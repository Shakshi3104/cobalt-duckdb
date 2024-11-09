import pandas as pd

from model.search.hybrid import HybridSearchClient
from model.data.notion_db import fetch_sakurap_corpus


if __name__ == "__main__":
    # Load dataset
    sakurap_df = fetch_sakurap_corpus("./data/sakurap_corpus.csv")
    # sakurap_df = pd.read_csv("./data/sakurap_corpus.csv")

    # hybrid search
    search_client = HybridSearchClient.from_dataframe(sakurap_df, "content")
    results = search_client.search_top_n("嵐 5人の歴史")
