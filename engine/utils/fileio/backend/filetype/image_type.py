from .base_type import BaseFileType
from engine.utils.logger import simple_logger


class ImageType(BaseFileType):
    __extension_name = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heif', 'heic', 'bpg', 'jp2', 'j2k', 'jpf', 'jpx', 'jpm', 'mj2', 'svg', 'svgz', 'ico', 'icns', 'cur', 'dds', 'tga', 'exr', 'hdr', 'pic', 'pnm', 'pbm', 'pgm', 'ppm', 'pam', 'pfm', 'pnm', 'sr', 'ras', 'jpe', 'jpge', 'jif', 'jfif', 'jfi', 'avif', 'avifs', 'apng', 'flif']
    __backends = ['PIL', 'cv2']
    __default_backend = 'PIL'
