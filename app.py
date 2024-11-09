import gradio as gr
import pandas as pd

from model.search.hybrid import HybridSearchClient
from model.data.notion_db import fetch_sakurap_corpus


def search(search_client: HybridSearchClient):
    def _search(query: str) -> pd.DataFrame:
        results = search_client.search_top_n(query)
        result = results[0]
        result["rank"] = result["rank"] + 1
        result = result[["rank", "title", "content", "rank_sparse", "rank_dense"]]
        result.columns = ["rank", "title", "rap lyric", "rank: surface", "rank: vector"]
        return result

    return _search


if __name__ == "__main__":
    # Load dataset
    sakurap_df = fetch_sakurap_corpus("./data/sakurap_corpus.csv")
    # Initialize search client
    search_client = HybridSearchClient.from_dataframe(sakurap_df, "content")

    with gr.Blocks() as search_interface:
        gr.Markdown("""
        # 💎 Cobalt
        Demo app for hybrid search with vector and surface search using [Ruri](https://huggingface.co/cl-nagoya/ruri-large), [BM25](https://github.com/dorianbrown/rank_bm25) and [Voyager](https://spotify.github.io/voyager/).
        """)
        # Input query
        search_query = gr.Textbox(label="Query", submit_btn=True)

        gr.Markdown("""
        ## Search Results

        """)
        # Search result
        result_table = gr.DataFrame(label="Result",
                                    column_widths=["5%", "20%", "65%", "5%", "5%"],
                                    wrap=True,
                                    datatype=["str", "str", "markdown", "str", "str"],
                                    interactive=False)

        # Event handler
        search_query.submit(fn=search(search_client), inputs=search_query, outputs=result_table)

    # App launch
    search_interface.queue()
    search_interface.launch(server_name="0.0.0.0")
