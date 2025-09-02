from abc import ABCMeta, abstractmethod

class BaseFileIO(metaclass=ABCMeta):
    """Base abstract class for fileio
    All fileio operation class should implement the methods
    declared in this class.
    """
    @property
    def filetype(self):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @staticmethod
    @abstractmethod
    def read_file(filepath):
        pass
    
    @staticmethod
    @abstractmethod
    def write_file(filepath, obj):
        pass
    
    
class BaseFileOperation:
    @staticmethod
    def copy_to(source_file, targte_file):
        pass

class Size:
    """A basic class for size unit convert
       
       You can simply call this class with Size(_int_number, '{UNIT}') to initialize,
       and call obj.unit to make convertion.
    """
    # basic unit defination
    B: float; b: float
    KB: float; kb: float
    MB: float; mb: float
    GB: float; gb: float
    TB: float; tb: float
    PB: float; pb: float
    EB: float; eb: float
    ZB: float; zb: float
    YB: float; yb: float
    # basic unit format defination
    UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    UNIT_MAP = {unit: 1024 ** idx for idx, unit in enumerate(UNITS)}
    
    def __init__(self, size, format, default_format: str=None):
        format = format.upper()
        if format not in self.UNIT_MAP:
            raise ValueError(f"Unsupported format '{format}'. Supported formats are: {', '.join(self.UNIT_MAP.keys())}")
        self.size_in_bytes = size * self.UNIT_MAP[format]
        self.default_format = default_format if default_format is not None else format
    
    def to(self, target_format: str):
        if target_format not in self.UNIT_MAP:
            raise ValueError(f"Unsupported target format '{target_format}'. Supported formats are: {', '.join(self.UNIT_MAP.keys())}")
        converted_size = self.size_in_bytes / self.UNIT_MAP[target_format]
        return converted_size
    
    @classmethod
    def convert(cls, size, current_format, target_format):
        instance = cls(size, current_format)
        return instance.to(target_format)
    
    def __call__(self):
        return self.size_in_bytes / self.UNIT_MAP[self.default_format]

    def __getattr__(self, name):
        name = name.upper()
        if name in self.UNITS:
            return self.to(name)
        raise AttributeError(f"'StorageConverter' object has no attribute '{name}'")
