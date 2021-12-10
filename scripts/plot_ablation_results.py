import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def load_data():
    path = pathlib.Path("../data/bim-gw.csv")
    data = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for k, row in enumerate(reader):
            if k >= 1:
                token = int(row[0].split("-")[1])
                if token >= 609:
                    if row[9] == '':
                        row[9] = np.inf
                    if row[10] == '':
                        row[10] = '0'
                    data.append({
                        "n_images": float(row[5]) * 500_000,
                        "sup_coef": float(row[6]),
                        "demicycle_coef": float(row[7]),
                        "cycle_coef": float(row[8]),
                        "z_size": int(row[4]),
                        "sup_loss": float(row[9]),
                        "nepochs": int(float(row[10])),
                        "id": token
                    })
    return data


def get_x_y(d, cond):
    kept_data = []
    labels = []
    job_ids = []
    for row in d:
        if cond(row):
            kept_data.append(row["sup_loss"])
            labels.append(row["n_images"])
            job_ids.append(row["id"])
    if len(kept_data):
        labels, kept_data, job_ids = zip(*sorted(zip(labels, kept_data, job_ids), key=lambda e: e[0]))
    return labels, kept_data, job_ids


if __name__ == '__main__':
    data = load_data()
    alpha = 1
    min_epoch = 90
    # z_size = 4

    for z_size in [4, 8, 10]:
        x, y, ids = get_x_y(data, lambda row: (row["sup_coef"] == alpha and row["demicycle_coef"] == 1
                                               and row["cycle_coef"] == 1 and row["nepochs"] >= min_epoch
                                               and row["z_size"] == z_size))
        plt.semilogx(x, y, label=f"All 3 losses ($\\alpha={alpha}$, z={z_size})")

        # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == 1 and row["demicycle_coef"] == 0
        #                                      and row["cycle_coef"] == 0 and row["nepochs"] >= min_epoch))
        # plt.semilogx(x, y, label="Supervision only")

        x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == alpha and row["demicycle_coef"] == 0
                                             and row["cycle_coef"] == 1 and row["nepochs"] >= min_epoch
                                             and row["z_size"] == z_size))
        plt.semilogx(x, y, label=f"Supervision + cycles ($\\alpha={alpha}$, z={z_size})")

    # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == alpha and row["demicycle_coef"] == 1
    #                                      and row["cycle_coef"] == 0 and row["nepochs"] >= min_epoch))
    # plt.semilogx(x, y, label=f"Supervision + demi-cycles ($\\alpha={alpha}$)")
    #
    # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == 0 and row["demicycle_coef"] == 0
    #                                      and row["cycle_coef"] == 1 and row["nepochs"] >= min_epoch))
    # plt.axhline(y, c="m", label=f"Cycle only")
    #
    # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == 0 and row["demicycle_coef"] == 1
    #                                      and row["cycle_coef"] == 1 and row["nepochs"] >= min_epoch))
    # plt.axhline(y, c="y", label=f"Demi and full cycle")

    plt.title("Effect of the three losses")
    plt.xlabel("Number of synchronized examples")
    plt.ylabel("Average Translation loss on the validation set")
    plt.legend()
    plt.show()
    print("ok")
    # loss_s_c = []
    # loss_cycle = [1.06, 0.659, 0.561, 0.282, 0.166, 0.0158, 0.0138]
    # loss_cycle_x = [0, 5, 10, 50, 100, 500, 1000]
    # loss_no_cycle = [0.816, 0.183, 0.126]
    # loss_no_cycle_x = [100, 500, 1000]
    # loss_supervision = [0.248, 0.0915, 0.0642, 0.0119, 0.00535, 0.0014, 0.00133, 0.0012]
    # loss_supervision_x = [100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000]
    #
    # plt.semilogx(loss_cycle_x, loss_cycle, label="All 3 losses ($\\alpha=10$)")
    # plt.semilogx(loss_no_cycle_x, loss_no_cycle, label="Demi-cycle + supervision ($\\alpha=10$)")
    # plt.semilogx(loss_supervision_x, loss_supervision, label="Only supervision loss")
    # plt.title("Effect of the three losses")
    # plt.xlabel("Number of synchronized examples")
    # plt.ylabel("Average Translation loss on the validation set")
    # plt.legend()
    # plt.show()
    #
