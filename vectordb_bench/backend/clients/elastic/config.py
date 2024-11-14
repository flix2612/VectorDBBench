from ..api import DBConfig

class ElasticConfig(DBConfig):
    host: str = "localhost"
    port: int = 9200

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port
        }