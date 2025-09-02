from .baseio import BaseFileIO, Size
from ..filetype import ImageType

import pathlib
from pathlib import Path
from typing import Union, List

# Imgage backend
import PIL
import cv2
import torchvision
from PIL import Image
from torch import Tensor

class ImageIO(BaseFileIO):
    __filetype = ImageType
    
    def __init__(self, backend: str=None):
        """
        Image fileio class
        Args:
            backend (str): The backend to read the image file.
                           currently supported backends are:
                           'PIL'(default), 'cv2', 'torchvision'
        """
        self.default_backend = ImageType._ImageType__default_backend
        self.backend = backend
        if backend == None:
            self.backend = self.default_backend
        if backend == 'torchvision':
            raise UserWarning(
                'Not recommended to use torchvision.'
            )

    @staticmethod
    def read_file(filepath: Path, backend: str='PIL'):
        if backend == 'PIL':
            return Image.open(filepath)
        elif backend == 'cv2':
            return cv2.imread(str(filepath))
        elif backend == 'torchvision':
            return torchvision.io.read_image(str(filepath))
        
    @staticmethod
    def write_file(filepath: Path, obj, backend: str='PIL'):
        if backend == 'PIL':
            obj.save(filepath)
        elif backend == 'cv2':
            cv2.imwrite(str(filepath), obj)
        elif backend == 'torchvision':
            torchvision.io.write_image(str(filepath), obj)
    
    def _read_file(self, filepath: Path):
        """
        Read image from file use the specified backend.
        Args:
            filepath (Path): The path to the image file.
        """
        return self.read_file(filepath, self.backend)
    
    def _write_file(self, filepath, obj):
        """
        Write image to file use the specified backend.
        Args:
            filepath (Path): The path to the image file.
            obj: The image object to write.
        """
        self.write_file(filepath, obj, self.backend)
    
    def _get_size(self, obj):
        """
        Get the size of the image.
        Args:
            obj: The image object.
        """
        if self.backend == 'PIL':
            return obj.size
        elif self.backend == 'cv2':
            return obj.shape[:2]
        elif self.backend == 'torchvision':
            return obj.shape[1:]
    
    def _convert_mode_pillow(self, obj, mode: str):
        assert mode is not None, "Read mode of the image must be specified."
        try:
            obj = obj.convert(mode)
        except Exception as e:
            raise RuntimeError('Error converting image to mode: {}'.format(mode))
        return obj
    
    def _convert_mode_cv2(self, obj, mode: str):
        map_dict = dict(
            L=cv2.COLOR_BGR2GRAY,
            RGB=cv2.COLOR_BGR2RGB,
            BGR=cv2.COLOR_RGB2BGR,
            RGBA=cv2.COLOR_BGR2RGBA,
            BGRA=cv2.COLOR_RGB2BGRA,
        )
        assert mode in map_dict, "Unsupported mode: {}".format(mode)
        try:
            obj = cv2.cvtColor(obj, map_dict[mode])
        except Exception as e:
            raise RuntimeError('Error converting image to mode: {}'.format(mode))
        return obj
    
    def read_image(self, filepath: Path, mode: str):
        if self.backend == 'PIL':
            obj = self._read_file(filepath)
            obj = self._convert_mode_pillow(obj, mode)
        elif self.backend == 'cv2':
            obj = self._read_file(filepath)
            obj = self._convert_mode_cv2(obj, mode)
        elif self.backend == 'torchvision':
            obj = self._read_file(filepath)
            obj = obj.permute(1, 2, 0)
        return obj

    def tensor_to_image(self, tensor: Tensor, save_path: Path):
        im = tensor.cpu().clone()
        im = im.squeeze(0)
        tensor2pil = torchvision.transforms.ToPILImage()
        im = tensor2pil(im)
        im.save(save_path)
        
    def verify_filename(self, name: str) -> bool:
        file_ext = pathlib.Path(name).suffix.split('.')[-1]
        extension_name = self.__filetype._ImageType__extension_name
        if file_ext in extension_name:
            return True
        return False

    def list_dir_image(self, path: Union[Path, str]) -> List[str]:
        if type(path) == str:
            path = Path(path)
        all_image_files = sorted(list(path.glob('**/*')))
        full_image_paths = []
        for file in all_image_files:
            if file.is_file() and self.verify_filename(file):
                full_image_paths.append(path / file)
        return full_image_paths
