import logging
from contextlib import contextmanager
from typing import Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .config import ElasticIndexConfig
from ..api import VectorDB

for logger in ("elastic", "elastic_transport"):
    logging.getLogger(logger).setLevel(logging.WARNING)

log = logging.getLogger(__name__)


class Elastic(VectorDB):
    def __init__(self,
                 dim,
                 db_config,
                 db_case_config: ElasticIndexConfig,
                 drop_old: bool = False):

        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_name = "vdb_bench_index"
        self.id_col_name = "id"
        self.vector_col_name = "embedding"

        host = db_config.get("host")
        port = db_config.get("port")
        self.connection_url = f"http://{host}:{port}"

        client = Elasticsearch(self.connection_url)
        log.info(f"Connected to {client.info()['name']}")

        if drop_old:
            index_exists = client.indices.exists(index=self.index_name)
            if index_exists:
                client.indices.delete(index=self.index_name)
                log.info(f"Deleted old index {self.index_name}")
            self._create_indice(client)

    @contextmanager
    def init(self) -> None:
        self.client = Elasticsearch(self.connection_url)
        log.info(f"Connected to {self.client.info()['name']}")

        yield
        self.client = None
        del (self.client)

    def _create_indice(self, client) -> None:
        mappings = {
            "properties": {
                self.id_col_name: {"type": "integer", "store": True},
                self.vector_col_name: {
                    "dims": self.dim,
                    **self.case_config.index_param()}
            }
        }
        try:
            client.indices.create(index=self.index_name, mappings=mappings)
            log.info(f"Created index {self.index_name}")
        except Exception as e:
            log.error(f"Failed creating index {self.index_name}: {e}")
            raise e

    def insert_embeddings(
            self,
            embeddings: Iterable[list[float]],
            metadata: list[int],
            **kwargs,
    ) -> (int, Exception):
        def _gen_data():
            for i, embedding in enumerate(embeddings):
                yield {
                    "_index": self.index_name,
                    "_source": {
                        self.id_col_name: metadata[i],
                        self.vector_col_name: embedding
                    }
                }
        try:
            bulk_insert_res = bulk(self.client, _gen_data())
            log.info(f"Inserted {bulk_insert_res} embeddings")
            return bulk_insert_res[0], None
        except Exception as e:
            log.warning(f"Failed to insert data: {self.index_name} error: {str(e)}")
            raise e

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
