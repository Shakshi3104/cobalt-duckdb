from copy import deepcopy
from typing import List, Union

import pandas as pd
import numpy as np

from loguru import logger
from tqdm import tqdm

from rank_bm25 import BM25Okapi

from model.search.base import BaseSearchClient
from model.utils.tokenizer import MeCabTokenizer


class BM25Wrapper(BM25Okapi):
    def __init__(self, dataset: pd.DataFrame, target, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.dataset = dataset
        corpus = dataset[target].values.tolist()
        super().__init__(corpus, tokenizer)

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]

        result = deepcopy(self.dataset.iloc[top_n])
        result["score"] = scores[top_n]
        return result


class BM25SearchClient(BaseSearchClient):
    def __init__(self, _model: BM25Okapi, _corpus: List[List[str]]):
        """

        Parameters
        ----------
        _model:
            BM25Okapi
        _corpus:
            List[List[str]], 検索対象の分かち書き後のフィールド
        """
        self.model = _model
        self.corpus = _corpus

    @staticmethod
    def tokenize_ja(_text: List[str]):
        """MeCab日本語分かち書きによるコーパス作成

        Args:
            _text (List[str]): コーパス文のリスト

        Returns:
            List[List[str]]: 分かち書きされたテキストのリスト
        """

        # MeCabで分かち書き
        parser = MeCabTokenizer.from_tagger("-Owakati")

        corpus = []
        with tqdm(_text) as pbar:
            for i, t in enumerate(pbar):
                try:
                    # 分かち書きをする
                    corpus.append(parser.parse(t).split())
                except TypeError as e:
                    if not isinstance(t, str):
                        logger.info(f"🚦 [BM25SearchClient] Corpus index of {i} is not instance of String.")
                        corpus.append(["[UNKNOWN]"])
                    else:
                        raise e
        return corpus

    @classmethod
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

        logger.info("🚦 [BM25SearchClient] Initialize from DataFrame")

        search_field = _data[_target]
        corpus = search_field.values.tolist()

        # 分かち書きをする
        corpus_tokenized = cls.tokenize_ja(corpus)
        _data["tokenized"] = corpus_tokenized

        bm25 = BM25Wrapper(_data, "tokenized")
        return cls(bm25, corpus_tokenized)

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

        logger.info(f"🚦 [BM25SearchClient] Search top {n} | {_query}")

        # 型チェック
        if isinstance(_query, str):
            _query = [_query]

        # クエリを分かち書き
        query_tokens = self.tokenize_ja(_query)

        # ランキングtop-nをクエリ毎に取得
        result = []
        for query in tqdm(query_tokens):
            df_res = self.model.get_top_n(query, self.corpus, n)
            # ランク
            df_res["rank"] = deepcopy(df_res.reset_index()).index
            df_res = df_res.drop(columns=["tokenized"])
            result.append(df_res)

        logger.success(f"🚦 [BM25SearchClient] Executed")

        return result
