from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio
from signalrcore.hub_connection_builder import HubConnectionBuilder

from .core.auth import AuthMixin
from .core.encryption import EncryptionMixin
from .core.file_utils import FileUtilsMixin
from .core.http_utils import HttpUtilsMixin
from .core.logger import LoggerMixin
from .core.metadata import MetadataMixin

from .modules.alias import AliasMixin
from .modules.datastructure import DataStructureMixin
from .modules.item import ItemMixin
from .modules.kvp import KeyValueMixin
from .modules.predicate import PredicateMixin
from .modules.relation import RelationMixin

from .helpers.mlflow_helpers import MlflowHelper
from .helpers.generic_helpers import GenericHelpers


class ManoloClient(
    AuthMixin,
    EncryptionMixin,
    FileUtilsMixin,
    HttpUtilsMixin,
    LoggerMixin,
    MetadataMixin,
    AliasMixin,
    DataStructureMixin,
    ItemMixin,
    KeyValueMixin,
    PredicateMixin,
    RelationMixin,
    MlflowHelper,
    GenericHelpers
):
    def __init__(self, base_url: str, username: str, password: str,
                 key: bytes = None, logging_level: logging = logging.INFO,
                 log_dir: str = "logs", log_to_file: bool = True,
                 log_to_console: bool = False, performance_monitor: bool = False, log_args: bool = False,
                 signalr_url: str = "/signalr", max_workers: int = 10):
        """
        Main Constructor for the Manolo Client

        Args:
            base_url (str): The base url of the Manolo API
            username (str): The username for the Manolo API
            password (str): The password for the Manolo API
            key (bytes): The key for the encryption/decryption of data (16, 24, or 32 bytes)
            logging_level (logging): The logging level for the client
            log_dir (str): The directory for the logs
            log_to_file (bool): Whether to log to file
            log_to_console (bool): Whether to log to console
            performance_monitor (bool): Whether to monitor performance
            log_args (bool): Whether to log arguments(args and kwargs might be large and expose sensitive information)

        """
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.key = key if key else None
        self._pending_items = asyncio.Queue()
        self.performance_monitor = performance_monitor

        # Initialize logger
        self.logger: logging.Logger = self.setup_logger(
            self.__class__.__name__, logging_level, log_dir, log_to_file, log_to_console)

        # Initialize SignalR connection
        self._signalr_connected = False
        self._completed_downloads = set()
        self._download_errors = []
        self.hub_connection = HubConnectionBuilder() \
            .with_url(f"{self.base_url}{signalr_url}") \
            .build()

        # Initialize QUIC session ticket
        self._quic_tickets = []

        self._tasks = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        super().__init__()

        if self.performance_monitor:
            self.decorate_methods_with_performance(log_args=log_args)
