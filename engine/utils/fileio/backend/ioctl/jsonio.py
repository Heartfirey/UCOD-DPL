from .baseio import BaseFileIO
from ..filetype import JSONType

import json

class JSONIO(BaseFileIO):
    """A class to handle reading and writing json files.
    """
    __filetype = JSONType()
    
    @staticmethod
    def read_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_file(filepath, obj):
        with open(filepath, 'w') as f:
            json.dump(obj, f)
