"""
Cache management utilities for dataset features and labels.
"""
import os
from typing import Optional, Any, List
from engine.utils.fileio.backend import MetaListPickleIO


class CacheManager:
    """Manages cache operations for dataset features and labels."""
    
    def __init__(self, base_path: str, logger=None):
        self.base_path = base_path
        self.logger = logger
        self._io = None
    
    @property
    def io(self) -> MetaListPickleIO:
        """Lazy initialization of cache IO."""
        if self._io is None:
            self._io = MetaListPickleIO(base_path=self.base_path, logger_in=self.logger)
        return self._io
    
    @property
    def mode(self) -> str:
        """Get cache mode."""
        return self.io.mode
    
    def dump_list(self, data_list: List[Any]) -> None:
        """Dump list of data to cache."""
        self.io.dump_list(data_list)
        self.io.reload_path()
    
    def read_file(self, index: int) -> Any:
        """Read cached file by index."""
        return self.io.read_file(index)
    
    def length(self) -> int:
        """Get cache length."""
        return self.io.len()


class MultiCacheManager:
    """Manages multiple cache instances for different data types."""
    
    def __init__(self, cache_dir: str, feature_extractor_type: str, mode: str, dataset_name: str, logger=None):
        self.cache_dir = cache_dir
        self.feature_extractor_type = feature_extractor_type
        self.mode = mode
        self.dataset_name = dataset_name
        self.logger = logger
        self._caches = {}
    
    def get_cache(self, cache_type: str) -> CacheManager:
        """Get or create cache manager for specific type."""
        cache_key = f"{cache_type}"
        
        if cache_key not in self._caches:
            if cache_type == 'features':
                cache_name = 'features_cache'
            else:
                cache_name = f"{cache_type}_cache"
            if cache_type == 'pseudo_label':
                cache_path = os.path.join(
                    self.cache_dir, 
                    cache_name,
                    self.dataset_name
                )
            else:
                cache_path = os.path.join(
                    self.cache_dir, 
                    cache_name,
                    self.feature_extractor_type, 
                    self.mode, 
                    self.dataset_name
                )
            self._caches[cache_key] = CacheManager(cache_path, self.logger)
        
        return self._caches[cache_key]
    
    def get_features_cache(self) -> CacheManager:
        """Get features cache manager."""
        return self.get_cache('features')
    
    def get_pseudo_label_cache(self) -> Optional[CacheManager]:
        """Get pseudo label cache manager (only for train mode)."""
        if self.mode == 'train':
            return self.get_cache('pseudo_label')
        return None
    
    def get_patch_cache(self) -> CacheManager:
        """Get patch cache manager."""
        return self.get_cache('patch')
    
    def get_m_patch_cache(self) -> CacheManager:
        """Get m-patch cache manager."""
        return self.get_cache('m_patch')
