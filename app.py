import gradio as gr
import pandas as pd

from model.search.vector import RuriVoyagerSearchClient
from model.data.notion_db import fetch_sakurap_corpus


def search(search_client: RuriVoyagerSearchClient):
    def _search(query: str) -> pd.DataFrame:
        results = search_client.search_top_n(query)
        result = results[0]
        result["rank"] = result["rank"] + 1
        result = result[["rank", "title", "content", "score"]]
        result.columns = ["rank", "title", "rap lyric", "distance"]
        return result

    return _search


if __name__ == "__main__":
    # Load dataset
    sakurap_df = fetch_sakurap_corpus("./data/sakurap_corpus.csv")
    # Initialize search client
    search_client = RuriVoyagerSearchClient.from_dataframe(sakurap_df, "content")

    with gr.Blocks() as search_interface:
        gr.Markdown("""
        # 💎 Cobalt DuckDB 🦆
        Demo app for vector search using [Ruri](https://huggingface.co/cl-nagoya/ruri-large) and DuckDB.

        You can search ARASHI's songs with rap lyrics by Sho Sakurai.
        """)
        # Input query
        search_query = gr.Textbox(label="Sakurap Words", submit_btn=True)

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