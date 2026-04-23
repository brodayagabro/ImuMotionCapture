from abc import ABC, abstractmethod
from typing import Callable

class BaseDataReader(ABC):
    def __init__(self, data_callback: Callable, error_callback: Callable):
        self.data_callback = data_callback
        self.error_callback = error_callback
        self.running = False

    @abstractmethod
    def connect(self) -> None: ...
    
    @abstractmethod
    def disconnect(self) -> None: ...
    
    @abstractmethod
    def read_loop(self) -> None: ...
    
    def send_command(self, command: str) -> None:
        pass