import abc
from typing import Optional

import MeCab
# from janome.tokenizer import Tokenizer


class BaseTokenizer:
    @abc.abstractmethod
    def parse(self, _text: str) -> str:
        """
        分かち書きした結果を返す

        Parameters
        ----------
        _text:
            str, 入力文章

        Returns
        -------
        parsed:
            str, 分かち書き後の文章, スペース区切り
        """
        raise NotImplementedError


class MeCabTokenizer(BaseTokenizer):
    def __init__(self, _parser: MeCab.Tagger) -> None:
        self.parser = _parser

    @classmethod
    def from_tagger(cls, _tagger: Optional[str]):
        parser = MeCab.Tagger(_tagger)
        return cls(parser)

    def parse(self, _text: str):
        return self.parser.parse(_text)


# class JanomeTokenizer(BaseTokenizer):
#     def __init__(self, _tokenizer: Tokenizer):
#         self.tokenizer = _tokenizer
#
#     @classmethod
#     def from_user_simple_dictionary(cls, _dict_filepath: Optional[str] = None):
#         """
#         簡易辞書フォーマットによるユーザー辞書によるイニシャライザー
#
#         https://mocobeta.github.io/janome/#v0-2-7
#
#         Parameters
#         ----------
#         _dict_filepath:
#             str, 簡易辞書フォーマットで書かれたユーザー辞書 (CSVファイル)のファイルパス
#         """
#
#         if _dict_filepath is None:
#             return cls(Tokenizer())
#         else:
#             return cls(Tokenizer(udic=_dict_filepath, udic_type='simpledic'))
#
#     def parse(self, _text: str) -> str:
#         return " ".join(list(self.tokenizer.tokenize(_text, wakati=True)))
