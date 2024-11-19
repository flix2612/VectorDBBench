from enum import Enum

from vectordb_bench.base import BaseModel
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType


class ElasticConfig(DBConfig):
    host: str = "localhost"
    port: int = 9200

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port
        }

class ElasticIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.ES_HNSW

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        params = {
            "type": "dense_vector",
            "index": True,
            "element_type": "float",
            "similarity": self.parse_metric(),
            "index_options": {
                "type": self.index.value,
                "m": self.M,
                "ef_construction": self.efConstruction,
            },
        }
        return params

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }