from langchain_core.stores import BaseStore
import shelve
from typing import Sequence, Optional, Union, Iterator, Tuple, Any

class LocalStore(BaseStore[str, Any]):
    """Local store implementation using shelve for persistent storage."""

    def __init__(self, directory: str) -> None:
        """Initialize the local store with a directory for storage."""
        self.directory = directory

    def mget(self, keys: Sequence[str]) -> list[Optional[Any]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        with shelve.open(self.directory) as db:
            return [db.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Any]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, Any]]): A sequence of key-value pairs.
        """
        with shelve.open(self.directory) as db:
            for key, value in key_value_pairs:
                db[key] = value

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        with shelve.open(self.directory) as db:
            for key in keys:
                if key in db:
                    del db[key]

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Yields:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        with shelve.open(self.directory) as db:
            for key in db.keys():
                if prefix is None or key.startswith(prefix):
                    yield key