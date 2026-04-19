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
    # df["RelGap_rsp"] = (df["pi_bf"] - df["pi_rsp"]) / df["pi_bf"]
    # df["RelGap_rsp2sp"] = (df["pi_sp"] - df["pi_rsp"]) / df["pi_sp"]
    # df["RelGap_ro2op"] = (df["pi_bf"] - df["pi_ro"]) / df["pi_bf"]

    orders = [
        '$N=15$\n$B=2$',     # "base"
        '$N$ ↑ 25\n$B=2$',   # "vary N"
        '$N=15$\n$B$ ↑ 4'    # "vary B"
    ]

    # F_list = ["GumBel", "NegExp", "UniForm", "NorMal", "BiNormal"]
    # F_legends = ["Gumbel", "NegExp", "Uniform", "Normal", "MixNormal"]

    # # the cases to show in the boxplot
    # problem_types_list = ["base", "N", "K"]
    # problem_types_labels = ["$N=15$\n$B=2$", "$N$ \u2191 25\n$B=2$", "$N=15$\n$B$ \u2191 4"]

    # Plot by correlation type
    for cor_type in ["ind", "linear"]:

        df_sub = df[df["cor"] == cor_type].copy()

        # print(df_sub["p_type"].unique())

        ####################################################
        ############### SP suboptimality gap ###############
        ####################################################

        # Create a figure and axis object
        fig, ax = plt.subplots()
        
        # plt.figure(figsize=(8, 6))
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

        # plt.show()
        # plt.savefig(results_folder + f"boxplot_all_{cor}.png", dpi=300)
        plt.savefig(folder + f"boxplot_SP2OP_{cor_type}.pdf", format="pdf")
        plt.close()