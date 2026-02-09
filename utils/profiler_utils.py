import torch
import torch.cuda.nvtx as nvtx
import functools
from contextlib import contextmanager
from omegaconf import DictConfig
from utils import comm


class Profiler:
    def __init__(self, cfg: DictConfig):
        self.enabled = cfg.profiler.enabled 
        self.profile_ranks = cfg.profiler.profile_ranks 
        self.profile_epoch = cfg.profiler.profile_epoch 
        self.current_rank = comm.get_world_rank()
        self.current_epoch = 0
        self.is_profiling = False

    def should_profile(self):
        """Check if NVTX ranges should be pushed (based on enabled and profile_ranks)"""
        return self.enabled and self.current_rank in self.profile_ranks

    def should_start_cuda_profiler(self):
        """Check if CUDA profiler should be started (based on enabled, profile_ranks, and profile_epoch)"""
        return self.should_profile() and self.current_epoch == self.profile_epoch

    def set_epoch(self, epoch):
        """Set the current epoch number and handle CUDA profiler start/stop"""
        # If we were profiling the previous epoch, stop profiling
        if self.is_profiling:
            torch.cuda.profiler.stop()
            self.is_profiling = False
            
        self.current_epoch = epoch
        
        # If this is the epoch we want to profile, start profiling 
        # controlled by -c flag in nsys
        if self.should_start_cuda_profiler():
            torch.cuda.profiler.start()
            self.is_profiling = True

    @contextmanager
    def profile_range(self, name):
        """Context manager for profiling code blocks"""
        if not self.should_profile():
            yield
            return
            
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()

class NoOpProfiler:
    enabled = False

    def should_profile(self):
        return False

    def set_epoch(self, _):
        pass

    @contextmanager
    def profile_range(self, name):
        yield

# Global profiler instance
_profiler = None

def init_profiler(cfg: DictConfig):
    """Initialize the global profiler instance"""
    global _profiler
    _profiler = Profiler(cfg)
    return _profiler

def get_profiler():
    """Get the global profiler instance"""
    if _profiler is None:
        return NoOpProfiler()
    return _profiler

def profile(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with get_profiler().profile_range(name or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def profile_range(name):
    """Convenience context manager using the global profiler"""
    with get_profiler().profile_range(name):
        yield 
