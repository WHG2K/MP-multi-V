import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == "__main__":

    # Read data (replace with your file path)
    # df = pd.read_json("./data/MP/RSP2SP/RSP2SP_all.jsonl", lines=True)
    folder_path = "./data/RSP2SP/"
    # folder_path = "./data/MP/RSP2SP2RO/instances/eta_0.2/"
    # df = pd.read_json(folder_path + "RSP2SP2RO_all_eta_0.2.jsonl", lines=True)
    df = pd.read_json(folder_path + "RSP2SP_data_solved.jsonl", lines=True)

    # Column needed for the boxplot; if 'RelGap' doesn't exist yet, compute it from two columns
    df["RelGap_rsp2sp"] = (df["pi_sp"] - df["pi_rsp"]) / df["pi_sp"]

    F_list = ["GumBel", "NegExp", "UniForm", "NorMal", "BiNormal"]
    F_legends = ["Gumbel", "NegExp", "Uniform", "Normal", "MixNormal"]

    F_mapping = dict(zip(F_list, F_legends))
    df["F"] = df["F"].map(F_mapping)

    # # The cases to show in the boxplot
    # problem_types_list = ["base", "N", "K"]
    # problem_types_labels = ["$N=15$\n$B=2$", "$N$ \u2191 25\n$B=2$", "$N=15$\n$B$ \u2191 4"]

    # Group and plot
    for cor_val in ["ind", "linear"]:

        for B_val in [2, 4]:
        # for B_val in [1]:

            df_sub = df[(df['cor'] == cor_val) & (df['B'] == B_val)].copy()

            # if (B_val == 2) and (cor_val == "ind"):
            #     df_sub_sub = df_sub[df_sub["F"] == "NegExp"].copy()
            #     df_sub_sub.to_excel("data/MP/RSP2SP/check_NegExp_B_2_cor_ind.xlsx")

            # print(df_sub["p_type"].unique())

            ####################################################
            ############### SP suboptimality gap ###############
            ####################################################

            # Create a figure and axis object
            fig, ax = plt.subplots()
            
            # plt.figure(figsize=(8, 6))
            sns.boxplot(
                data = df_sub,
                x = "N",
                y = "RelGap_rsp2sp",
                hue = "F",
                showfliers = False
            )

            ax.set_title("")
            ax.set_xlabel("N")
            ax.set_ylabel('RelGap')

            ax.legend(title="")

            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            if cor_val == "ind":
                low, high = -0.02, 0.10
            else:
                low, high = -0.04, 0.10

            # Set padding ratio, e.g. 5%
            pad_ratio = 0.05
            pad = (high - low) * pad_ratio

            # Final lower and upper limits
            # ax.set_ylim(low - pad, high + pad)
            plt.ylim(low - pad, high + pad)

            # plt.show()
            # plt.savefig(results_folder + f"boxplot_all_{cor}.png", dpi=300)
            plt.savefig(folder_path + f"RSP2SP_B_{B_val}_{cor_val}.pdf", format="pdf", bbox_inches="tight")
            plt.close()