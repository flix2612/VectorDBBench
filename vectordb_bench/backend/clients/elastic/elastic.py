import logging
import time
from contextlib import contextmanager
from typing import Iterable
from ..api import VectorDB, DBCaseConfig
from .config import ElasticConfig

for logger in ("elastic", "elastic_transport"):
    logging.getLogger(logger).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

class Elastic(VectorDB):
    def __init__(
        self,
        **kwargs,
    ):
        pass


    @contextmanager
    def init(self) -> None:
        """connect to elastic"""
        pass

    def _create_indice(self, client) -> None:
        pass

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        """Insert the embeddings to the elastic."""
        pass

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[tuple[int, float]]: list of k most similar embeddings in (id, score) tuple to the query embedding.

        """
        pass



    def optimize(self):
        """optimize will be called between insertion and search in performance cases."""
        pass

    def ready_to_load(self):
        """ready_to_load will be called before load in load cases."""
        pass
