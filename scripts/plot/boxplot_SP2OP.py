import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == "__main__":

    # Read data (replace with your file path)
    folder = "./data/SP2OP/"
    df = pd.read_json(folder + "SP2OP_data_solved.jsonl", lines=True)

    # Create p_type column
    def assign_p_type(row):
        if row["N"] == 15 and row["B"] == 2:
            p_type = "base"
        elif row["N"] == 25:
            p_type = "N"
        elif row["B"] == 4:
            p_type = "B"
        else:
            raise ValueError(f"Unknown configuration: N={row['N']}, B={row['B']}")
        
        p_type_mapping = {
            "base": "$N=15$\n$B=2$",
            "N": "$N$ \u2191 25\n$B=2$",
            "B": "$N=15$\n$B$ \u2191 4"
        }
        return p_type_mapping[p_type]

    df["p_type"] = df.apply(assign_p_type, axis=1)

    # Compute RelGap columns from pairwise revenue comparisons
    df["RelGap_sp"] = (df["pi_bf"] - df["pi_sp"]) / df["pi_bf"]

    orders = [
        '$N=15$\n$B=2$',     # "base"
        '$N$ ↑ 25\n$B=2$',   # "vary N"
        '$N=15$\n$B$ ↑ 4'    # "vary B"
    ]


    # # the cases to show in the boxplot

    # Plot by correlation type
    for cor_type in ["ind", "linear"]:

        df_sub = df[df["cor"] == cor_type].copy()


        ####################################################
        ############### SP suboptimality gap ###############
        ####################################################

        # Create a figure and axis object
        fig, ax = plt.subplots()
        
        sns.boxplot(
            data = df_sub,
            x = "p_type",
            y = "RelGap_sp",
            hue = "F",
            order = orders,
            showfliers = False
        )

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel('RelGap')

        ax.legend(title="")

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.savefig(folder + f"boxplot_SP2OP_{cor_type}.pdf", format="pdf")
        plt.close()