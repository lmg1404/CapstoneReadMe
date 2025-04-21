# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for tasks including how to move legends and debugging.
#   https://chat.openai.com/
#
# Additional Resources Consulted:
# - Stack Overflow (stackoverflow.com) for troubleshooting and syntax patterns
# - Matplotlib documentation (https://matplotlib.org/stable/index.html)
# - Streamlit documentation (https://docs.streamlit.io/)
# - pandas documentation (https://pandas.pydata.org/docs/)
# - Streamlit dashboard example (https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/#2-perform-eda-analysis)
# - Ask Science Direct AI dashboard example (https://www.sciencedirect.com/ai)
#
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils import get_pg_connection


def generate_figure2(start="2025-01-06", end="2025-03-30"):
    """
    Quick function to generate Figure 2 for our final report.
    Time range had to be adjusted to to be only the full weeks.
    Author: Sam
    input:
        start: start date to pull data from and visualize
        end: end date to pull data from and visualize
    returns:
    None
    """
    conn = get_pg_connection()
    with conn.cursor() as cur:
            cur.execute("""
                SELECT p.publishdate::date, jsonb_array_elements_text(p.tags) AS tag
                FROM papers p
                JOIN summaries s ON p.paper_id = s.paper_id
                WHERE p.publishdate BETWEEN %s AND %s;
            """, (start, end))
            rows = cur.fetchall()

    conn.close()

    df = pd.DataFrame(rows, columns=["date", "tag"])

    df= df.loc[df['tag'] != "Machine Learning"]

    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").apply(lambda r: r.start_time)
    
    #Uncomment for debugging
    #print(f"show {len(df)} data points between {start} and {end}")
    #print(df)

    tags = df["tag"].value_counts().nlargest(5).index
    df = df[df["tag"].isin(tags)]

    counts = df.groupby(["week", "tag"]).size()
    counts = counts.unstack(fill_value=0)
    total = df.groupby("week").size()

    plt.figure(figsize=(11, 5))
    plt.plot(total.index, total.values, label="All Topics", color="black", linewidth=2)



    #modifying this to change dash type as well as color so color-blind people can understand the graphic, staying in line with the accessiblity focus of our project
    
    tag_options = {
                        tags[0]: {"color": "#1f77b4", "linestyle": "dashed", "marker": "o"},
                        tags[1]: {"color": "#ff7f0e", "linestyle": "dashed", "marker": "s"}, 
                        tags[2]: {"color": "#2ca02c", "linestyle": "dashed", "marker": "D"},  
                        tags[3]: {"color": "#d62728", "linestyle": "dashed", "marker": "^"}, 
                        tags[4]: {"color": "#9467bd", "linestyle": "dashed", "marker": "X"}
                    }
    
    for tag in tags:
        style = tag_options.get(tag, {"color": "gray", "linestyle": "dashed", "marker": "o"})
        plt.plot(
            counts.index,
            counts[tag],
            label=tag,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4
        )


    plt.title("Weekly Summary Trends by 5 Most Frequent Topics", fontsize=16)
    plt.xlabel("Week", fontsize=12)
    plt.xticks(rotation=45) 

    plt.ylabel("Number of Summaries Generated", fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5)
    
    #moving legend outside of graph for clarity
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, title="Topics")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig("Figure2.png", dpi=300)
    plt.show()


generate_figure2()
