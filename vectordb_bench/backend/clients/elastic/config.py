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


class ElasticIndexConfig(BaseModel):
    index: IndexType
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"


class HNSWConfig(ElasticIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.ES_HNSW

    efConstruction: int | None = None
    m: int | None = None

    def index_param(self) -> dict:
        params = {
            "type": "dense_vector",
            "index": True,
            "element_type": "float",
            "similarity": self.parse_metric(),
            "index_options": {
                "type": self.index.value,
                "m": self.m,
                "ef_construction": self.efConstruction,
            },
        }
        return params

    def search_param(self) -> dict:
        pass


class FLATConfig(ElasticIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.ES_FLAT

    def index_param(self) -> dict:
        params = {
            "type": "dense_vector",
            "index": True,
            "element_type": "float",
            "similarity": self.parse_metric(),
            "index_options": {
                "type": self.index.value,
            },
        }
        return params

    def search_param(self) -> dict:
        pass


_elastic_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.ES_HNSW: HNSWConfig,
    IndexType.Flat: FLATConfig,
    IndexType.ES_FLAT: FLATConfig,
}
