import matplotlib.pyplot as plt
import numpy as np

from bim_gw.utils.neptune import CSVLog


def load_data():
    # data.append({
    #     "n_images": float(row[5]) * 500_000,
    #     "sup_coef": float(row[6]),
    #     "demicycle_coef": float(row[7]),
    #     "cycle_coef": float(row[8]),
    #     "z_size": int(row[4]),
    #     "sup_loss": float(row[9]),
    #     "nepochs": int(float(row[10])),
    #     "id": token
    # })
    log = CSVLog("../data/bim-gw.csv")
    return log.data


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
    log = CSVLog("../data/bim-gw.csv")
    log = log.add_token_column().filter_between("_token", 609, 846)
    log = log.add_column("n_images", lambda row: int(row["prop_lab_images"] * 500_000))

    alpha = 4
    min_epoch = 1
    seed = 0
    # z_size = 4
    log = log.filter_eq("parameters/seed", seed).filter_between('metrics/epoch (last)', min_epoch)
    log = log.filter_neq('metrics/val_supervision_loss (last)', '')

    for alpha in [2, 4, 10]:
        l = log.filter_eq("parameters/losses/coefs/supervision", alpha)
        for z_size in [12]:
            l = l.filter_eq('gw_z_size', z_size)

            cond = lambda row: row['parameters/losses/coefs/demi_cycles'] == 1 and row['parameters/losses/coefs/cycles'] == 1
            l = l.filter(cond)

            lf = l.filter_between('metrics/epoch (last)', min_epoch)
            x, y, ids = lf.zip(['n_images', 'metrics/val_supervision_loss (last)', '_token'], sort=lambda e: e[0])
            line = plt.loglog(x, y, label=f"All 3 losses ($\\alpha={alpha}$, z={z_size})")

            lf = l.filter_between('metrics/epoch (last)', 80)
            x, y = lf.zip(['n_images', 'metrics/val_supervision_loss (last)'], sort=lambda e: e[0])
            if alpha == 10:
                plt.loglog(x, y, "x", c=line[0].get_color(), label="Trained with at least 80 epochs.")
            else:
                plt.loglog(x, y, "x", c=line[0].get_color())

    plt.title("Effect of $\\alpha$")
    plt.xlabel("Number of synchronized examples")
    plt.ylabel("Average Translation loss on the validation set")
    plt.legend()
    plt.show()

    for alpha in [2]:
        l = log.filter_eq("parameters/losses/coefs/supervision", alpha)
        for z_size in [12]:
            l = l.filter_eq('gw_z_size', z_size)

            cond = lambda row: row['parameters/losses/coefs/demi_cycles'] == 1 and row['parameters/losses/coefs/cycles'] == 1
            l = l.filter(cond)

            lf = l.filter_between('metrics/epoch (last)', min_epoch)
            x, y, ids = lf.zip(['n_images', 'metrics/val_supervision_loss (last)', '_token'], sort=lambda e: e[0])
            line = plt.loglog(x, y, label=f"All 3 losses ($\\alpha={alpha}$, z={z_size})")

            lf = l.filter_between('metrics/epoch (last)', 80)
            x, y = lf.zip(['n_images', 'metrics/val_supervision_loss (last)'], sort=lambda e: e[0])
            if alpha == 10:
                plt.loglog(x, y, "x", c=line[0].get_color(), label="Trained with at least 80 epochs.")
            else:
                plt.loglog(x, y, "x", c=line[0].get_color())

    plt.title("Effect of $\\alpha$")
    plt.xlabel("Number of synchronized examples")
    plt.ylabel("Average Translation loss on the validation set")
    plt.legend()
    plt.show()

            # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == alpha and row["demicycle_coef"] == 0
            #                                      and row["cycle_coef"] == 1 and row["nepochs"] >= min_epoch
            #                                      and row["z_size"] == z_size))
            # line = plt.loglog(x, y, label=f"Supervision + cycles ($\\alpha={alpha}$, z={z_size})")
            #
            # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == alpha and row["demicycle_coef"] == 0
            #                                      and row["cycle_coef"] == 1 and row["nepochs"] >= 80
            #                                      and row["z_size"] == z_size))
            # plt.loglog(x, y, "x", c=line[0].get_color())

        # x, y, _ = get_x_y(data, lambda row: (row["sup_coef"] == 1 and row["demicycle_coef"] == 0
        #                                      and row["cycle_coef"] == 0 and row["nepochs"] >= min_epoch
        #                                      and row["z_size"] == 12))
        # line = plt.loglog(x, y, label="Supervision only")
        # x, y, ids = get_x_y(data, lambda row: (row["sup_coef"] == 1 and row["demicycle_coef"] == 0
        #                                        and row["cycle_coef"] == 0 and row["nepochs"] >= 80
        #                                        and row["z_size"] == 12))
        # plt.loglog(x, y, "x", c=line[0].get_color())

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
