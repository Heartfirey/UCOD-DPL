import os
import re
import logging
import multiprocessing
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
import prettytable as PTable
try:
    import torch.distributed as dist
except ImportError:
    dist = None
# time etc.
import pytz
import ntplib
from datetime import datetime
# logging.basicConfig(level=logging.INFO)
# DEFAULT FORMAT SETTINGS
# formats for components of the log record
FMT = dict(
    LEVEL=r'[%(levelname)8s]',
    TIME=r'%(asctime)s',
    INFO=r'[%(name)s] %(module)s.%(funcName)s%(lineno)4s',
    MSG=r'%(message)s')

# general date/time format
DATE_FMT = r'%y.%m.%d %H:%M'
TIME_ZONE = 'Asia/Shanghai'

# record definition for logging in file
FILE_FMT = ' '.join([fmt for fmt in FMT.values()])

class SingletonType(type):
    """ metaclass for create singleton class in multiprocess environment """
    _instances = dict()
    _lock = multiprocessing.Lock()
    
    def __call__(cls, name, *args, **kwargs):
        with cls._lock:
            if name not in cls._instances:
                # if the named instances has not been created, create it
                instance = super().__call__(name, *args, **kwargs)
                cls._instances[name] = instance
            else:
                print(f"[SKIP] Warning: Process ID {os.getpid()} is attempting to recreate the instance named '{name}'.")
            return cls._instances[name]

class TagStrippingFormatter(logging.Formatter):
    """ Custom formatter for stripping tags from the log message(used for file output) """
    def __init__(self, fmt: str=FMT, datefmt: str=DATE_FMT, style: str='%', validate: bool=True, timezone: str=TIME_ZONE):
        super().__init__(fmt, datefmt, style, validate)
        self.tag_pattern = re.compile(r'\[([\w\s]+?)(?: [^\]]*)?\](.*?)\[\/\1\]', re.DOTALL)
        self.timezone = pytz.timezone(timezone) if type(timezone) is str else timezone
        
    def formatTime(self, record, datefmt=DATE_FMT):
        ct = datetime.fromtimestamp(record.created, self.timezone)
        if datefmt:
            return ct.strftime(datefmt)
        else:
            return ct.strftime('%Y-%m-%d %H:%M:%S')
        
    def format(self, record):
        original = super().format(record)
        return re.sub(self.tag_pattern, r'\2', original)


class StreamTimeFormatter:
    """ Stream time formatter for rich handler """
    def __init__(self, fmt: str=DATE_FMT, timezone: str=TIME_ZONE):
        self.fmt = fmt
        self.timezone = pytz.timezone(timezone) if type(timezone) is str else timezone
    
    def __call__(self, *args):
        shanghai_time = args[0].astimezone(self.timezone)
        return shanghai_time.strftime(self.fmt)


class LoggerCreationError(Exception):
    """Custom error raised when creating logger failed"""
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg
    def __str__(self):
        return self.msg

class Logger(object, metaclass=SingletonType):
    def __init__(self, 
                 name: str, 
                 log_path: str=None, 
                 level: str='INFO', 
                 stream_level: str=None, 
                 file_level: str=None, 
                 multi_rank: list=[-1], 
                 timezone: str=TIME_ZONE,
                 timefmt: str=DATE_FMT):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        
        # calibrate the system time
        self.timezone = pytz.timezone(timezone)
        
        # multirank settings
        if type(multi_rank) is list and -1 not in multi_rank:
            self.multi_rank = multi_rank
            if len(multi_rank) > 1 and log_path is not None:
                if 'log' not in log_path:
                    raise LoggerCreationError("When using multi-rank logging, the log_path must contain 'log'.")
                log_path = log_path.replace('.log', f'_{os.getpid()}.log')
        else:
            self.multi_rank = None
        
        # detect the level for logger
        level = 'INFO' if level is None else level.upper()
        stream_level = stream_level if stream_level is not None else level
        file_level = file_level if file_level is not None else level
        stream_time_formatter = StreamTimeFormatter(fmt=timefmt, timezone=self.timezone)
        stream_handler = RichHandler(level=stream_level, markup=True, log_time_format=stream_time_formatter)
        if not any(isinstance(h, RichHandler) for h in self.logger.handlers):
            self.logger.addHandler(stream_handler)
        if log_path is not None:
            file_handler = RotatingFileHandler(log_path, maxBytes=1024*1024*10, backupCount=5)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(TagStrippingFormatter(fmt=FILE_FMT, datefmt=timefmt, timezone=self.timezone))
            if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
                self.logger.addHandler(file_handler)
        
        self.log(f"[bold green]Logger created on process {os.getpid()} with name '{name}'.[/bold green]")
        
    def __getattr__(self, name):
        if hasattr(self.__class__, name):
            return getattr(self.__class__, name)
        elif hasattr(self.logger, name):
            return getattr(self.logger, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def rank_judge(self):
        if self.multi_rank is None or \
            not dist.is_initialized() or\
            (dist.get_rank() in self.multi_rank):
            return True
        return False
            
    def log(self, *args, level: str='INFO', **kwargs):
        if type(level) is str:
            level = eval(f'logging.{level.upper()}')
        if self.rank_judge():
            self.logger.log(level, *args, **kwargs)
    
    def log_table(self, context: dict, name: str="Log Table", save_extra: str=None, level='INFO'):
        if type(level) is str:
            level = eval(f'logging.{level.upper()}')
        max_len = max([len(context[key]) for key in context.keys()])
        new_table = PTable.PrettyTable()
        new_table.field_names = context.keys()
        for i in range(max_len):
            new_table.add_row([str(context[key][i]) if i < len(context[key]) else '' for key in context.keys()])
        if self.rank_judge():
            self.logger.log(level, f'[bold green]{name}[/bold green]' + '\n' + new_table.get_string())
        if save_extra is not None:
            try:
                with open(save_extra, 'w') as f:
                    f.write(f'{name}\n' + new_table.get_string)
            except:
                self.logger.error(f"arg 'save_extra' has been set, but failed to save to '{save_extra}'!")

    def log_to_file(self, *args, level='INFO', **kwargs):
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.emit(logging.LogRecord(name=self.logger.name, level=logging.INFO, pathname='', func='FILELOG', lineno=0, msg=args[0], args=None, exc_info=None))
                break

def simple_logger(level: str='NOTSET') -> logging.Logger:
    logger = logging.getLogger('rich')
    if not logger.hasHandlers():
        stream_time_formatter = StreamTimeFormatter(fmt=DATE_FMT, timezone=TIME_ZONE)
        handler = RichHandler(level=level, rich_tracebacks=True, markup=True, log_time_format=stream_time_formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

if __name__ == "__main__":
    # import torch
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(dist.get_rank())
    
    logger = Logger('test', 'test.log', multi_rank=[0])
    logger.log('[bold green]test_info[/bold green]')
    logger.log('[bold green]debug_info[/bold green]',level="DEBUG")
    test_table = {
        "metric1": [111,222,333],
        'metric2': [444,555,666],
        'metric3': [777,888,999]
    }
    logger.log_table(test_table)
    
    simple_logger_test = simple_logger('INFO')
    simple_logger_test.exception("Test Error")

    logger.log('[bold green]test_info22222[/bold green]',level="ERROR")
