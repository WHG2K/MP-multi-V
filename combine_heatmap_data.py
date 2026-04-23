import os
import json

if __name__ == "__main__":

    path     = "./data/heatmaps/"
    out_file = "./data/heatmaps/HEATMAP_data.jsonl"

    configs = [
        {"beta": 0.1, "eta": 0.2, "cor": "ind"},
        {"beta": 0.1, "eta": 0.2, "cor": "linear"},
        {"beta": 0.1, "eta": 0.4, "cor": "ind"},
        {"beta": 0.1, "eta": 0.4, "cor": "linear"},
    ]

    count = 0
    with open(out_file, 'w') as fout:
        for cfg in configs:
            filename = f"hm_raw_beta-{cfg['beta']}_eta-{cfg['eta']}_{cfg['cor']}.jsonl"
            filepath = os.path.join(path, filename)

            with open(filepath, 'r') as fin:
                for line in fin:
                    s = line.strip()
                    if not s:
                        continue
                    inst = json.loads(s)

                    record = {
                        "N":    len(inst["u"]),
                        "beta": cfg["beta"],
                        "eta":  cfg["eta"],
                        "cor":  cfg["cor"],
                        "u":    inst["u"],
                        "r":    inst["r"],
                        "v":    inst["v"],
                    }
                    json.dump(record, fout)
                    fout.write("\n")
                    count += 1

    print(f"Combined {count} rows → {out_file}")