from typing import Union, List

import pandas as pd
from copy import deepcopy

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from model.search.base import BaseSearchClient
from model.search.surface import BM25SearchClient
from model.search.vector import RuriVoyagerSearchClient

from model.utils.timer import stop_watch


def reciprocal_rank_fusion(sparse: pd.DataFrame, dense: pd.DataFrame, k=60) -> pd.DataFrame:
    """
    Reciprocal Rank Fusionを計算する

    Notes
    ----------
    RRFの計算は以下の式

    .. math:: RRF = \sum_{i=1}^n \frac{1}{k+r_i}

    Parameters
    ----------
    sparse:
        pd.DataFrame, 表層検索の検索結果
    dense:
        pd.DataFrame, ベクトル検索の結果
    k:
        int,

    Returns
    -------
    rank_results:
        pd.DataFrame, RRFによるリランク結果

    """
    # カラム名を変更
    sparse = sparse.rename(columns={"rank": "rank_sparse"})
    dense = dense.rename(columns={"rank": "rank_dense"})
    # denseはランク以外を落として結合する
    dense_ = dense["rank_dense"]

    # 順位を1からスタートするようにする
    sparse["rank_sparse"] += 1
    dense_ += 1

    # 文書のインデックスをキーに結合する
    rank_results = pd.merge(sparse, dense_, how="left", left_index=True, right_index=True)

    # RRFスコアの計算
    rank_results["rrf_score"] = 1 / (rank_results["rank_dense"] + k) + 1 / (rank_results["rank_sparse"] + k)

    # RRFスコアのスコアが大きい順にソート
    rank_results = rank_results.sort_values(["rrf_score"], ascending=False)
    rank_results["rank"] = deepcopy(rank_results.reset_index()).index

    return rank_results


class HybridSearchClient(BaseSearchClient):
    def __init__(self, dense_model: BaseSearchClient, sparse_model: BaseSearchClient):
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    @classmethod
    @stop_watch
    def from_dataframe(cls, _data: pd.DataFrame, _target: str):
        """
        検索ドキュメントのpd.DataFrameから初期化する

        Parameters
        ----------
        _data:
            pd.DataFrame, 検索対象のDataFrame

        _target:
            str, 検索対象のカラム名

        Returns
        -------

        """
        # 表層検索の初期化
        dense_model = BM25SearchClient.from_dataframe(_data, _target)
        # ベクトル検索の初期化
        sparse_model = RuriVoyagerSearchClient.from_dataframe(_data, _target)

        return cls(dense_model, sparse_model)

    @stop_watch
    def search_top_n(self, _query: Union[List[str], str], n: int = 10) -> List[pd.DataFrame]:
        """
        クエリに対する検索結果をtop-n個取得する

        Parameters
        ----------
        _query:
            Union[List[str], str], 検索クエリ
        n:
            int, top-nの個数. デフォルト 10.

        Returns
        -------
        results:
            List[pd.DataFrame], ランキング結果
        """

        logger.info(f"🚦 [HybridSearchClient] Search top {n} | {_query}")

        # 型チェック
        if isinstance(_query, str):
            _query = [_query]

        # ランキングtop-nをクエリ毎に取得
        result = []
        for query in tqdm(_query):
            assert len(self.sparse_model.corpus) == len(
                self.dense_model.corpus), "The document counts do not match between sparse and dense!"

            # ドキュメント数
            doc_num = len(self.sparse_model.corpus)

            # 表層検索
            logger.info(f"🚦 [HybridSearchClient] run surface search ...")
            sparse_res = self.sparse_model.search_top_n(query, n=doc_num)
            # ベクトル検索
            logger.info(f"🚦 [HybridSearchClient] run vector search ...")
            dense_res = self.dense_model.search_top_n(query, n=doc_num)

            # RRFスコアの計算
            logger.info(f"🚦 [HybridSearchClient] compute RRF scores ...")
            rrf_res = reciprocal_rank_fusion(sparse_res[0], dense_res[0])

            # 結果をtop Nに絞る
            top_num = 10
            rrf_res = rrf_res.head(top_num)
            logger.info(f"🚦 [HybridSearchClient] return {top_num} results")

            result.append(rrf_res)

        return result
