import os
from pathlib import Path
import abc

import pandas as pd
from dotenv import load_dotenv

import notion_client as nt
from notion2md.exporter.block import StringExporter

from loguru import logger


class BaseNotionDatabase:
    """
    Notion DBからページのコンテンツを取り出すベースのクラス
    """
    def __init__(self):
        load_dotenv()
        self.notion_database_id = os.getenv("NOTION_DATABASE_ID")
        self.integration_token = os.getenv("INTEGRATION_TOKEN")

        # notion2mdの環境変数
        os.environ["NOTION_TOKEN"] = os.getenv("INTEGRATION_TOKEN")

        self.notion_client = nt.Client(auth=self.integration_token)

    def load_database(self) -> list[dict]:
        """
        Notion DBのページ一覧を取得

        Returns:

        """
        results = []
        has_more = True
        start_cursor = None

        while has_more:
            db = self.notion_client.databases.query(
                **{
                    "database_id": self.notion_database_id,
                    "start_cursor": start_cursor
                }
            )
            # 100件までしか1回に取得できない
            # 100件以上ある場合 has_more = True
            has_more = db["has_more"]
            # 次のカーソル
            start_cursor = db["next_cursor"]

            # 取得結果
            results += db["results"]

        return results

    @abc.abstractmethod
    def load_content(self) -> list[dict]:
        """
        Notion DBのページの中身をdictで返す
        Returns:

        """
        raise NotImplementedError


class SakurapDB(BaseNotionDatabase):
    def load_database(self) -> list[dict]:
        """
        Notion DBのページ一覧を取得

        Returns:
            results:
                list[dict]

        """
        results = []
        has_more = True
        start_cursor = None

        while has_more:
            # "Rap詞 : 櫻井翔"がTrueのもののみ取得
            db = self.notion_client.databases.query(
                **{
                    "database_id": self.notion_database_id,
                    "filter": {
                        "property": "Rap詞 : 櫻井翔",
                        "checkbox": {
                            "equals": True
                        }
                    },
                    "start_cursor": start_cursor
                }
            )
            # 100件までしか1回に取得できない
            # 100件以上ある場合 has_more = True
            has_more = db["has_more"]
            # 次のカーソル
            start_cursor = db["next_cursor"]

            # 取得結果
            results += db["results"]

        return results

    def __load_blocks(self, block_id: str) -> str:
        """
        Notionのページをプレーンテキストで取得する (Notion Official API)

        Parameters
        ----------
        block_id:
            str, Block ID

        Returns
        -------
        texts:
            str
        """
        block = self.notion_client.blocks.children.list(
            **{
                "block_id": block_id
            }
        )

        # プレーンテキストを繋げる
        def join_plain_texts():
            text = [blck["paragraph"]["rich_text"][0]["plain_text"] if len(blck["paragraph"]["rich_text"])
                    else "\n" for blck in block["results"]]

            texts = "\n".join(text)
            return texts

        return join_plain_texts()

    def load_content(self) -> list[dict]:
        """
        Notion DBのページの中身をdictで返す

        Returns:
            lyrics:
                list[dict]
        """

        # DBのページ一覧を取得
        db_results = self.load_database()
        logger.info("🚦 [Notion] load database...")

        # コンテンツ一覧
        lyrics = []

        logger.info("🚦 [Notion] start to load each page content ...")
        # 各ページの処理
        for result in db_results:
            block_id = result["id"]
            # rap_lyric = self.__load_blocks(block_id)

            # Markdown形式でページを取得
            rap_lyric = StringExporter(block_id=block_id).export()
            # Markdownの修飾子を削除
            rap_lyric = rap_lyric.replace("\n\n", "\n").replace("<br/>", "\n").replace("*", "")

            lyrics.append(
                {
                    "title": result["properties"]["名前"]["title"][0]["plain_text"],
                    "content": rap_lyric
                }
            )

        logger.info("🚦 [Notion] Finish to load.")

        return lyrics


def fetch_sakurap_corpus(filepath: str, refetch=False) -> pd.DataFrame:
    """
    サクラップのコーパスを取得する
    CSVファイルが存在しないときにNotionから取得する

    Parameters
    ----------
    filepath:
        str
    refetch:
        bool

    Returns
    -------

    """
    filepath = Path(filepath)

    if not filepath.exists() or refetch:
        # CSVファイルを保存するディレクトリが存在しなかったら作成する
        if not filepath.parent.exists():
            logger.info(f"🚦 [Notion] mkdir {str(filepath.parent)} ...")
            filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info("🚦 [Notion] fetch from Notion DB ...")
        # dictを取得
        rap_db = SakurapDB()
        lyrics = rap_db.load_content()

        lyrics_df = pd.DataFrame(lyrics)
        lyrics_df.to_csv(filepath, index=False)
    else:
        logger.info("🚦 [Notion] load CSV file.")

        lyrics_df = pd.read_csv(filepath)

    return lyrics_df


if __name__ == "__main__":
    sakurap_db = SakurapDB()
    lyrics = sakurap_db.load_content()
