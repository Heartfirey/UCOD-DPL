import re
from pathlib import Path

def natural_sort_key(s):
    if type(s) is Path:
        s = str(s.name)
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def convert_path(path) -> Path:
    if type(path) is str:
        return Path(path)
    elif isinstance(path, Path):
        return path
    raise ValueError('Invalid path type: {}'.format(type(path)))
