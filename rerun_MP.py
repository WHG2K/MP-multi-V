import json
from tqdm import tqdm
from src.models import MPMVSurrogate
from src.utils import get_distribution_from_name


if __name__ == "__main__":

    filename = 'MP_data_test.jsonl'

    with open(filename, 'r', encoding='utf-8') as f:
        total_instances = sum(1 for _ in f)

    with open(filename, 'r', encoding='utf-8') as f:

        for line in tqdm(f, total=total_instances, desc="Redo SP"):
            
            inst = json.loads(line)
            N = inst['N']
            u = inst['u']
            r = inst['r']
            v = inst['v']
            n_pick = inst['B']
            distr = get_distribution_from_name(inst['F'])
            C = tuple(inst['C'])

            sp = MPMVSurrogate(u, r, v, n_pick, distr, C)
            x = sp.solve(method='SP', n_steps=1001)

            print("sp solution: ", x)
            print("bf solution: ", inst["bf_x"])

            # tqdm.write("see if I can print")

            # break



            # tqdm.write(f"  get dict: {inst}")

