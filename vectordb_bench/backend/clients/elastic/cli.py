import logging
from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run, \
    HNSWBaseRequiredTypedDict

log = logging.getLogger(__name__)


class ElasticTypesDict(CommonTypedDict):
    host: Annotated[
        str, click.option("--host", type=str, help="Elasticsearch host address", required=True)
    ]
    port: Annotated[
        int, click.option("--port", type=int, help="Elasticsearch port number", required=True)
    ]


class ElasticHNSWTypedDict(ElasticTypesDict, HNSWBaseRequiredTypedDict):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(ElasticHNSWTypedDict)
def ElasticHNSW(**parameters: Unpack[ElasticHNSWTypedDict]):
    from .config import ElasticConfig, ElasticIndexConfig

    run(
        db=DB.Elastic,
        db_config=ElasticConfig(
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=ElasticIndexConfig(
            efConstruction=parameters["ef_construction"],
            m=parameters["m"],
        ),
        **parameters,
    )
