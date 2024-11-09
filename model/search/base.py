import abc
from typing import List, Union

import pandas as pd


class BaseSearchClient:
    """
    検査インタフェースクラス
    """
    corpus: pd.DataFrame | list | None = None

    @classmethod
    @abc.abstractmethod
    def from_dataframe(cls, _data: pd.DataFrame, _target: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def search_top_n(self, _query: Union[List[str], str], n: int=10) -> List[pd.DataFrame]:
        raise NotImplementedError()