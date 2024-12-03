"""Wrapper around the Milvus vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager
from typing import Iterable

from pymilvus import Collection, utility, CollectionSchema, DataType, FieldSchema, MilvusException, connections

from .config import MilvusIndexConfig
from ..api import VectorDB

log = logging.getLogger(__name__)


class Milvus(VectorDB):
    col: Collection | None = None

    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: MilvusIndexConfig,
            collection_name: str = "VectorDBBenchCollection",
            drop_old: bool = False,
            name: str = "Milvus",
            **kwargs,
    ):
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        self._primary_field = "pk"
        self._scalar_field = "id"
        self._vector_field = "vector"
        self._index_name = "vector_idx"

        connections.connect(**self.db_config, timeout=30)
        if drop_old and utility.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(self._primary_field, DataType.INT64, is_primary=True),
                FieldSchema(self._scalar_field, DataType.INT64),
                FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)
            ]

            log.info(f"{self.name} create collection: {self.collection_name}")

            col = Collection(
                name=self.collection_name,
                schema=CollectionSchema(fields),
                consistency_level="Session",
            )

            col.create_index(
                self._vector_field,
                self.case_config.index_param(),
                index_name=self._index_name,
            )

        connections.disconnect("default")

    @contextmanager
    def init(self) -> None:
        connections.connect(**self.db_config, timeout=60)
        self.col = Collection(self.collection_name)

        yield
        connections.disconnect("default")

    def _optimize(self):
        self._post_insert()
        log.info(f"{self.name} optimizing before search")
        try:
            self.col.load()
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        try:
            self.col.flush()
            # wait for index done and load refresh
            self.col.create_index(
                self._vector_field,
                self.case_config.index_param(),
                index_name=self._index_name,
            )

            utility.wait_for_index_building_complete(self.collection_name)

            def wait_index():
                while True:
                    progress = utility.index_building_progress(self.collection_name)
                    if progress.get("pending_index_rows", -1) == 0:
                        break
                    time.sleep(5)

            wait_index()

            try:
                self.col.compact()
                self.col.wait_for_compaction_completed()
            except Exception as e:
                log.warning(f"{self.name} compact error: {e}")
                if hasattr(e, 'code'):
                    if e.code().name == 'PERMISSION_DENIED':
                        log.warning(f"Skip compact due to permission denied.")
                        pass
                else:
                    raise e
            wait_index()
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def ready_to_load(self):
        raise NotImplementedError("Not used for Benchmarking")

    def optimize(self):
        assert self.col, "Please call self.init() before"
        self._optimize()

    def insert_embeddings(
            self,
            embeddings: Iterable[list[float]],
            metadata: list[int],
            **kwargs,
    ) -> (int, Exception):
        assert self.col is not None
        assert len(embeddings) == len(metadata)

        insert_count = 0
        try:
            log.info(f"Inserting {len(embeddings)} embeddings")
            insert_data = [metadata, metadata, embeddings]
            res = self.col.insert(insert_data)
            insert_count = len(res.primary_keys)
        except MilvusException as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def search_embedding(
            self,
            query: list[float],
            k: int = 100,
            filters: dict | None = None,
            timeout: int | None = None,
    ) -> list[int]:
        assert self.col is not None

        res = self.col.search(
            data=[query],
            anns_field=self._vector_field,
            param=self.case_config.search_param(),
            limit=k,
        )

        ret = [result.id for result in res[0]]
        return ret
