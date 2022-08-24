import os
import numpy as np


def main():
    dir_results = "./results_warwick/f1-check"
    files = os.listdir(dir_results)
    for f in files:
        metric = 'f1'
        if 'res-f1' in f:
            clf_name = f[7:].split('.')[0]
        else:
            metric = 'mcc'
            clf_name = f[8:].split('.')[0]
        measures = []
        with open(os.path.join(dir_results, f), 'r') as fp:
            for res in fp:
                measures.append(float(res.replace('\n','').replace(',','.')))

        measures = np.array(measures)

        print(f"{metric} {clf_name} Train avg = {measures[:85].mean():.4f}, Test avg = {measures[85:].mean():.4f}")


if __name__ == "__main__":
    main()