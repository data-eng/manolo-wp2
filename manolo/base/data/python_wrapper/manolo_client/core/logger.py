from functools import wraps
import logging
import sys
import os
from datetime import datetime
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client import ManoloClient


class LoggerMixin:
    @staticmethod
    def setup_logger(
        name: str = "manolo_datatier_client",
        level=logging.INFO,
        log_dir: str = "logs",
        log_to_file: bool = True,
        log_to_console: bool = False,
    ) -> logging.Logger:
        logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def log_performance(self: "ManoloClient", logger: logging.Logger, log_args: bool = False):
        """
        Log the performance of a method and its arguments if log_args is True

        Args:
            logger (logging.Logger): The logger to use for logging
            log_args (bool): Whether to log the arguments of the method
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if log_args:
                    arg_list = [repr(a) for a in args]
                    kwarg_list = [f"{k}={v!r}" for k, v in kwargs.items()]
                    all_args = arg_list + kwarg_list
                    arg_summary = ", ".join(all_args)
                    if len(arg_summary) > 100:
                        arg_summary = arg_summary[:97] + "..."
                    extra = f"({arg_summary})"
                else:
                    extra = ""

                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000

                method_name = func.__qualname__.split('.')[-1]

                logger.info(
                    f"Performance: {method_name}{extra} took {elapsed_ms:.3f} ms"
                )
                return result
            return wrapper
        return decorator

    def decorate_methods_with_performance(self: "ManoloClient", log_args: bool = False):
        """
        Decorate all methods in the class with the log_performance decorator
        """
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name == 'log_performance':
                continue
            attr = getattr(self, attr_name)
            if callable(attr):
                decorated = self.log_performance(self.logger, log_args)(attr)
                setattr(self, attr_name, decorated)
