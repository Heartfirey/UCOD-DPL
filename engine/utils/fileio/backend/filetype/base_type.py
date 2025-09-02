from abc import ABCMeta, abstractmethod
from typing import Union, List

class BaseFileType(metaclass=ABCMeta):
    """Base abstract filetype class
    All specific filetype should implent the methods declared
    in this class.
    """
    __extension_name: Union[str, List]=None
    @property
    def extension_name(self):
        return self.__extension_name
    
    def match_name(self, name: str) -> bool:
        if type(self.extension_name) is str:
            return name.split('.')[-1] == self.extension_name
        if type(self.extension_name) is list:
            match = False
            for ext in self.extension_name:
                if name.split('.')[-1] == ext:
                    match = True; break
            return match
    
    # @abstractmethod
    # def parse_file(self):
    #     pass
