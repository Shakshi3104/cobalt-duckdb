import os

from typing import List, Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from copy import deepcopy
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

import sentence_transformers as st

import duckdb

from model.search.base import BaseSearchClient
from model.utils.timer import stop_watch


def array_to_string(array: np.ndarray) -> str:
    """
    np.ndarrayを文字列に変換する

    Parameters
    ----------
    array:
        np.ndarray

    Returns
    -------
    array_string:
        str
    """
    array_string = f"{array.tolist()}"
    return array_string


class RuriEmbedder:
    def __init__(self, model: Optional[st.SentenceTransformer] = None):

        load_dotenv()

        # モデルの保存先
        self.model_dir = Path("models/ruri")
        model_filepath = self.model_dir / "ruri-large"

        # モデル
        if model is None:
            if model_filepath.exists():
                logger.info(f"🚦 [RuriEmbedder] load ruri-large from local path: {model_filepath}")
                self.model = st.SentenceTransformer(str(model_filepath))
            else:
                logger.info(f"🚦 [RuriEmbedder] load ruri-large from HuggingFace🤗")
                token = os.getenv("HF_TOKEN")
                self.model = st.SentenceTransformer("cl-nagoya/ruri-large", token=token)
                # モデルを保存する
                logger.info(f"🚦 [RuriEmbedder] save model ...")
                self.model.save(str(model_filepath))
        else:
            self.model = model

    def embed(self, text: Union[str, list[str]]) -> np.ndarray:
        """

        Parameters
        ----------
        text:
            Union[str, list[str]], ベクトル化する文字列

        Returns
        -------
        embedding:
             np.ndarray, 埋め込み表現. トークンサイズ 1024
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding


class RuriVoyagerSearchClient(BaseSearchClient):
    def __init__(self, dataset: pd.DataFrame, target: str,
                 vector_store_name: str,
                 model: RuriEmbedder):
        load_dotenv()
        # オリジナルのコーパス
        self.dataset = dataset
        self.corpus = dataset[target].values.tolist()

        # 埋め込みモデル
        self.embedder = model

        # DuckDBのテーブル名
        self.vector_store_name = vector_store_name

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
        logger.info("🚦 [RuriVoyagerSearchClient] Initialize from DataFrame")

        search_field = _data[_target]
        corpus = search_field.values.tolist()

        # 埋め込みモデルの初期化
        embedder = RuriEmbedder()

        # Ruriの前処理
        corpus = [f"文章: {c}" for c in corpus]

        # ベクトル化する
        embeddings = embedder.embed(corpus)

        # 埋め込みベクトルの次元
        num_dim = embeddings.shape[1]
        logger.debug(f"🚦⚓️ [RuriVoyagerSearchClient] Number of dimensions of Embedding vector is {num_dim}")

        # DuckDBに挿入
        vector_store_name = "ruri_vector_index"
        vdb = pd.DataFrame({"index": range(len(embeddings)), "embedding": embeddings.tolist()})
        duckdb.register(vector_store_name, vdb)

        return cls(_data, _target, vector_store_name,embedder)

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

        logger.info(f"🚦 [RuriVoyagerSearchClient] Search top {n} | {_query}")

        # 型チェック
        if isinstance(_query, str):
            _query = [_query]

        # Ruriの前処理
        _query = [f"クエリ: {q}" for q in _query]

        # ベクトル化
        embeddings_queries = self.embedder.embed(_query)

        # ランキングtop-nをクエリ毎に取得
        result = []
        for embeddings_query in tqdm(embeddings_queries):
            num_dim = len(embeddings_query)
            distance = duckdb.sql(f"""
            select
                index, 
                array_cosine_distance(embedding::DOUBLE[{num_dim}], {embeddings_query.tolist()}::DOUBLE[{num_dim}]) as distance
            from {self.vector_store_name}
            order by distance
            limit {n}
            """).df()

            # 類似度スコア
            df_res = deepcopy(self.dataset.iloc[distance["index"].tolist()])
            df_res["score"] = distance["distance"].tolist()
            # ランク
            df_res["rank"] = deepcopy(df_res.reset_index()).index

            result.append(df_res)

        return result
