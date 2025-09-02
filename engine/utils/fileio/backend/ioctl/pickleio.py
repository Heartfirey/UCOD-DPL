from .baseio import BaseFileIO, Size
from .jsonio import JSONIO
from ..filetype import PickleType, TorchPickleType
from .utils import natural_sort_key, convert_path

import os
import pickle
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Union

from engine.utils.logger import Logger


try:
    import torch
except ImportError:
    torch = None
    
    logger.log('torch is not installed, torch related functions are disabled', level='ERROR')

class PickleIO(BaseFileIO):
    __filetype = PickleType
    
    @staticmethod
    def read_file(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
    @staticmethod
    def write_file(filepath, obj):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    

class TorchPickleIO(BaseFileIO):
    __filetype = TorchPickleType
    
    def __init__(self):
        assert torch is not None, 'torch is not installed, torch related functions are disabled'
    
    @staticmethod
    def read_file(filepath: Path, map_location='cpu'):
        with open(filepath, 'rb') as f:
            return torch.load(f, map_location=map_location)
    
    @staticmethod
    def write_file(filepath, obj):
        with open(filepath, 'wb') as f:
            torch.save(obj, f)
            
class MetaListPickleIO(BaseFileIO):
    """
    A class to handle reading and writing list of objects using pickle files.
    """
    def __init__(
        self,
        index_path: Union[Path, str]=None,
        base_path: Union[Path, str]=None,
        file_prefix: str='data',
        logger_in=None
    ):
        if index_path is not None:
            self.index_path = convert_path(index_path)
            self.base_path = self.index_path.parent
        elif base_path is not None:
            self.base_path = convert_path(base_path)
            self.index_path = self.base_path / 'index.json'
        else:
            raise ValueError('Either index_path or base_path must be specified.')
        self.file_prefix = file_prefix
        self.prefix_counter = {}

        self.logger = logger_in
        if self.logger is None:
            self.logger = Logger('test', multi_rank=[0])
            
        self.logger.log('Checking integrity of cache files...', level='INFO')
        check_results, fb = self.check_integrity(self.index_path)
        if not check_results:
            self.logger.log('[yellow bold]Cache not available or corrupted, woring on writing mode[/yellow bold]', level='WARNING')
            self.mode = 'w'
        else:
            self.logger.log('[green bold]Cache files are available and valid, working on reading mode[/green bold]', level='INFO')
            self.mode = 'r'
        
        self.index_map = dict()
        if self.mode == 'r':
            self._prepare_reading()
        
    @staticmethod
    def check_integrity(index_file_path: Optional[Union[str, Path]]) -> bool:
        index_file_path: Path = convert_path(index_file_path)
        if not index_file_path.exists():
            return False, 'Index file does not exist.'
        index_map = JSONIO.read_file(index_file_path)
        for index, file in index_map.items():
            if not (index_file_path.parent / file).exists():
                return False, 'File with index {} does not exist.'.format(index)
        return True, '_' 
    
    def reload_path(self):
        self.logger.log('Reloading and checking the integrity of cache files...', level='INFO')
        check_results, fb = self.check_integrity(self.index_path)
        if not check_results:
            self.logger.log('[yellow bold]Cache not available or corrupted, switching to writing mode[/yellow bold]', level='WARNING')
            self.mode = 'w'
            self.index_map = dict()  
        else:
            self.logger.log('[green bold]Cache files are valid, switching to reading mode[/green bold]', level='INFO')
            self.mode = 'r'
            self._prepare_reading()  

    def _prepare_reading(self):
        self.index_map = JSONIO.read_file(self.index_path)
        for index, file in self.index_map.items():
            self.index_map[index] = self.base_path / file

    def read_file(self, index):
        assert self.mode == 'r', 'Not working on read mode!'
        return PickleIO.read_file(self.index_map[str(index)])
    
    def len(self):
        return len(self.index_map)
    
    def write_file(self, index, obj, file_name=None):
        assert self.mode == 'w', 'Not working on write mode!'
        if file_name:
            self.index_map[index] = '{}_{}.pkl'.format(file_name, self.prefix_counter.get(file_name, 0))
            self.prefix_counter[file_name] = self.prefix_counter.get(file_name, 0) + 1 
        else:
            self.index_map[index] = '{}_{}.pkl'.format(self.file_prefix, index)
        PickleIO.write_file(self.base_path / self.index_map[index], obj)
    
    def dump_list(self, obj_list, file_name_list=None):
        for index, obj in enumerate(obj_list):
            file_name = file_name_list[index] if file_name_list else None
            self.write_file(index, obj, file_name)

        JSONIO.write_file(self.index_path, self.index_map)

        
    
    
class ChunkPickleIO(BaseFileIO):
    """A class to handle reading and writing large iterables using chunked pickle files.
    
    Modes:
        - 'write': Split and save the iterable into chunked pickle files.
        - 'read': Read and iterate over the data from chunked pickle files with preloading.

    """
    __filetype = PickleType
    
    def __init__(
        self,
        mode: str='w',
        index_path: Path=None,
        chunk_size: Size=Size(1, 'GB'),
        preload_window_size: int=2,
    ):
        assert mode in ('r', 'w'), "Mode must be 'read' or 'write"

        self.mode = mode
        self.index_path = index_path if type(index_path) is Path else Path(index_path)
        self.base_path = self.index_path.parent
        self.chunk_size = chunk_size
        self.preload_window_size = preload_window_size
        
        self.current_chunk_index = 0
        self.index_map = dict()
        self.buffer = Queue()
        self.read_thread = None
        self.stop_event = threading.Event()

        if self.mode == 'read':
            assert self.base_path is not None, 'Directory must be specified in read mode.'
            
    def __del__(self):
        self.stop_event.set()
        if self.read_thread:
            self.read_thread.join()
    
    def _prepare_reading(self):
        self.index_map = JSONIO.read_file(self.index_path)
        
        self.file_paths = sorted(self.index_map.values(), key=natural_sort_key)
        self.read_thread = threading.Thread(target=self._preload_window)
        self.read_thread.start()
    
    def _preload_window(self):
        while not self.stop_event.is_set():
            if self.buffer.qsize() < self.window_size and self.current_chunk_index < len(self.file_paths):
                try:
                    chunk = PickleIO.read_file(self.base_path / self.file_paths[self.current_chunk_index])
                    for item in chunk:
                        self.buffer.put(item)
                    self.current_chunk_index += 1
                except Exception as e:
                    logger.error(f'Error reading chunk {self.current_chunk_index}: {e}')
                    exit(-1)
            else:
                self.stop_event.wait(1)
    
    def read(self):
        #TODO: implement the chunk load logic
        pass
    
    def write(self):
        #TODO: implement the chunk write logic
        pass
    

