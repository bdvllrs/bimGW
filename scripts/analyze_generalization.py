import numpy as np
from matplotlib import pyplot as plt

from bim_gw.utils.csv import CSVLog

if __name__ == '__main__':
    log = CSVLog("../data/bim-gw.csv")
    # log = log.add_token_column().filter_between("_token", 910, 958)  # attributes
    # log = log.add_token_column().filter_between("_token", 959, 1007)  # BERT vectors
    log = log.add_token_column().filter_between("_token", 1273, 1357)  # BERT vectors
    log = log.filter_neq("prop_lab_images", "")
    log = log.add_column("n_images", lambda row: int(row["prop_lab_images"] * 500_000))
    # log = log.filter_eq("parameters/seed", 0)
    seeds = [0, 1, 2, 3, 4]
    log = log.filter_in("parameters/seed", seeds)

    n_images = sorted(np.unique(log.values("n_images")))

    for loss in ["loss"]:
        for type_ in ["in_dist", "ood"]:
            for supervision, use_cycles in [("supervised", 0), ("cycles", 1)]:
                x = []
                y = []
                err = []
                n_runs = []
                for n_image in n_images:
                    l = log.filter_eq("n_images", n_image)
                    l = l.filter_eq("parameters/losses/coefs/cycles", use_cycles)
                    x.append(n_image)
                    values = np.array(l.values(f'training/val_{type_}_supervision_{loss} (last)'))
                    # if values.shape[0] != len(seeds):
                    #     print("nooo")
                    # assert values.shape[0] == len(seeds)
                    print(values.shape[0], n_image / 500_000)
                    n_runs.append(values.shape[0])
                    y.append(np.mean(values))
                    err.append(np.std(values))

                plt.errorbar(x, y, yerr=np.divide(np.array(err), np.sqrt(n_runs)), label=f"{type_} {supervision}")
                # plt.plot(x, y, label=f"{type_} {supervision}")

        plt.gca().set_yscale("log")
        plt.gca().set_xscale("log")
        plt.xlabel("Number of synchronized images")
        plt.ylabel(f"AVG translation validation losses")
        plt.legend()
        plt.title("AVG translation losses. Each point is averaged over 5 trials.")
        plt.show()
