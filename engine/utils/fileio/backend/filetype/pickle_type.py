from .base_type import BaseFileType
from engine.utils.logger import simple_logger
try:
    import pickle
except ImportError as err:
    logger = simple_logger("INFO")
    logger.exception('Unable to import pickle!')

class PickleType(BaseFileType):
    __extension_name = ['pkl']

class TorchPickleType(BaseFileType):
    __extension_name = ['pt', 'pth']
    
class JoblibType(BaseFileType):
    __extension_name = ['joblib']
    

