import pytest

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.models.hirth3203.params import MVEMParams


@pytest.fixture
def params() -> MVEMParams:
    return MVEMParams()


@pytest.fixture
def engine(params: MVEMParams) -> Hirth3203Engine:
    return Hirth3203Engine(params=params)
