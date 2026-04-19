import json
from tqdm import tqdm
from src.models_new import MixtureMNL, MPMVSurrogate, MixtureSP, MixedSP
from src.distributions import GumBel
from src.heuristics import ADXOPT, RO, Greedy
import argparse
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import time
from skopt import forest_minimize, gp_minimize, plots
from skopt.space import Real
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    parser.add_argument("--data", type=str, default="", help='data file name')

    return parser.parse_args()


class OBJ_FUNC_SP:

    def __init__(self, sp_model):
        self.sp_model = sp_model

    def __call__(self, w):
        return -self.sp_model.SP(w)[1]


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # choose environment
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')
        
    filename = args.data

    with open(filename, 'r', encoding='utf-8') as f:
        total_instances = sum(1 for _ in f)

    updated_instances = []

    with open(filename, 'r', encoding='utf-8') as f:

        for line in tqdm(f, total=total_instances, desc="Redo SP"):
            
            inst = json.loads(line)
            N = inst['N']
            K = inst['K']
            pf = inst['pf']
            r = inst['r']
            weights = inst['weights']
            C = tuple(inst['C'])

            mixmnl = MixtureMNL(pf, r, weights)
            time_1 = time.time()
            # ro = RO(N)
            # x_ro, val_ro = ro.maximize(mixmnl, C=C, r=r)
            time_2 = time.time()
            # adxopt = ADXOPT(N)
            # x_adxopt, val_adxopt = adxopt.maximize(mixmnl, C=C, verbose=0)
            time_3 = time.time()
            # greedy = Greedy(N)
            # x_greedy, val_greedy = greedy.maximize(mixmnl, C=C)
            time_4 = time.time()
            x_milp, val_milp = mixmnl.solve(C=C)
            time_5 = time.time()
            u_mix = np.log(pf).tolist()
            # models = []
            # distr = GumBel()
            # for k in range(K):
            #     u = u_mix[k]
            #     models.append(MPMVSurrogate(u, r, [0.0], 1, distr, C))
            # mixSP = MixtureSP(models, weights)
            # adxopt = ADXOPT(N)
            # x_sp_adxopt, val_sp_adxopt = adxopt.maximize(mixSP, C=C, r=r)
            time_6 = time.time()
            # y = np.ones(N, dtype=int)
            # w = np.zeros(K, dtype=float)
            # for i in range(K):
            #     w[i] = models[i]._w_x_(x_adxopt)
            # w = w.tolist()
            # print(x_adxopt, mixmnl(x_adxopt))
            mixedSP = MixedSP(u_mix, r, [[0.0] for _ in range(K)], 1, GumBel(), C, weights)
            # x_mixedSP, val_mixedSP = mixedSP.SP(w)
            # print(x_mixedSP, val_mixedSP)
            # x_bayesian, val_bayesian = mixSP(w)
            low, high = mixedSP._get_box_range()
            space = [Real(low, high) for _ in range(K)]
            obj_func_sp = OBJ_FUNC_SP(mixedSP)
            # w_sp_ao = [0.0 for _ in range(K)]
            # for i in range(K):
            #     w_sp_ao[i] = models[i]._w_x_(x_sp_adxopt)
            # print(obj_func_sp(w_sp_ao))
            # res = forest_minimize(
            #     func=obj_func_sp,
            #     dimensions=space,
            #     # x0 = [w_sp_ao],
            #     n_calls=5000,
            #     n_initial_points=1000,  
            #     base_estimator="ET",   # Extremely Randomized Trees
            #     acq_func="EI",         # Expected Improvement
            #     random_state=0,        
            #     n_jobs=24
            # )
            res = gp_minimize(
                func=obj_func_sp,
                dimensions=space,
                n_calls=200,            # total number of evaluations (down from 5000)
                n_initial_points=20,     # initial random samples (down from 1000)
                acq_func="EI",           # Expected Improvement
                random_state=0,
                n_jobs=24
            )
            w_baye = res.x
            x_sp_baye = mixedSP.SP(w_baye)[0]
            x_sp_baye = [int(round(x)) for x in x_sp_baye]
            # print("===========================================================================")
            # print("最佳目标值：", res.fun)
            # print("最佳参数：")
            # for name, val in zip([d.name for d in space], res.x):
            #     print(f"  {name} = {val:.6f}")
            # # ─── 7. 可视化（可选）───────────────────
            # # 7.1 采集函数曲线
            # plots.plot_gaussian_process(res)
            # plt.title("GP surrogate & acquisition (EI)")
            # plt.show()
            # # 7.2 参数—目标平面（pairwise）
            # plots.plot_objective(res)
            # plt.show()
            # print("===========================================================================")
            time_7 = time.time()
            # print("time bayesian:", time_7 - time_6)


            # inst["x_ro"] = x_ro
            # inst["x_adxopt"] = x_adxopt
            # inst["x_greedy"] = x_greedy
            inst["x_milp"] = x_milp
            # inst["x_sp_adxopt"] = x_sp_adxopt
            inst["x_baye"] = x_sp_baye

            # inst["pi_ro"] = mixmnl(x_ro)
            # inst["pi_adxopt"] = mixmnl(x_adxopt)
            # inst["pi_greedy"] = mixmnl(x_greedy)
            inst["pi_milp"] = mixmnl(x_milp)
            # inst["pi_sp_adxopt"] = mixmnl(x_sp_adxopt)
            inst["pi_baye"] = mixmnl(x_sp_baye)

            # inst["time_ro"] = time_2 - time_1
            # inst["time_adxopt"] = time_3 - time_2
            # inst["time_greedy"] = time_4 - time_3
            inst["time_milp"] = time_5 - time_4
            # inst["time_sp_adxopt"] = time_6 - time_5
            inst["time_baye"] = time_7 - time_6

            updated_instances.append(inst)

            # break
            # if (ct > 10):
                # break


    root, ext = os.path.splitext(filename)
    output_path = f"{root}_cardinality_solved{ext}"

    updated_df = pd.DataFrame(updated_instances)
    updated_df.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False,
    )
