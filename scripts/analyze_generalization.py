import numpy as np
from matplotlib import pyplot as plt

from bim_gw.utils.csv import CSVLog

if __name__ == '__main__':
    log = CSVLog("../data/bim-gw.csv")
    # log = log.add_token_column().filter_between("_token", 910, 958)  # attributes
    log = log.add_token_column().filter_between("_token", 959, 1007)  # BERT vectors
    log = log.add_column("n_images", lambda row: int(row["prop_lab_images"] * 500_000))

    n_images = sorted(np.unique(log.values("n_images")))

    for loss in ["v_to_t_0", "t_to_v_0"]:
        for type_ in ["in_dist", "ood"]:
            for supervision, use_cycles in [("supervised", 0), ("cycles", 1)]:
                x = []
                y = []
                err = []
                for n_image in n_images:
                    l = log.filter_eq("n_images", n_image)
                    l = l.filter_eq("parameters/losses/coefs/cycles", use_cycles)
                    x.append(n_image)
                    values = np.array(l.values(f'metrics/val_{type_}_loss_supervision_{loss} (last)'))
                    y.append(np.mean(values))
                    err.append(np.std(values))

                plt.errorbar(x, y, yerr=np.array(err) / np.sqrt(5), label=f"{type_} {supervision}")

        # plt.gca().set_yscale("log")
        plt.gca().set_xscale("log")
        plt.xlabel("Number of synchronized images")
        plt.ylabel(f"AVG translation validation {loss} losses")
        plt.legend()
        plt.title("AVG translation losses. Each point is averaged over 5 trials.")
        plt.show()
