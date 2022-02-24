import matplotlib.pyplot as plt

from bim_gw.utils.csv import CSVLog


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


def cond_losses(*, demi_cycle, cycle, supervision):
    def aux(row):
        return (
            ((row['parameters/losses/coefs/demi_cycles'] == 0 and not demi_cycle) or
            (row['parameters/losses/coefs/demi_cycles'] != 0 and demi_cycle)) and

            ((row['parameters/losses/coefs/cycles'] == 0 and not cycle) or
             (row['parameters/losses/coefs/cycles'] != 0 and cycle)) and

            ((row['parameters/losses/coefs/supervision'] == 0 and not supervision) or
             (row['parameters/losses/coefs/supervision'] != 0 and supervision))
        )
    return aux


if __name__ == '__main__':
    log = CSVLog("../data/bim-gw.csv")
    log = log.add_token_column()
    # log = log.filter_between("_token", 609, 846)
    log = log.filter_between("_token", 1045, 1097)
    log = log.add_column("n_images", lambda row: int(row["prop_lab_images"] * 500_000))

    alpha = 4
    min_epoch = 30
    seed = 42
    losses = {"loss": ""}
    # z_size = 4
    log = log.filter_eq("parameters/seed", seed).filter_between('metrics/epoch (last)', min_epoch)
    log = log.filter_eq('gw_z_size', 12)

    loss_parts = {
        # "Demi-cycles": cond_losses(demi_cycle=True, cycle=False, supervision=False),
        # "Complete cycles": cond_losses(demi_cycle=False, cycle=True, supervision=False),
        "Supervision": cond_losses(demi_cycle=False, cycle=False, supervision=True),
        # "Cycles": cond_losses(demi_cycle=True, cycle=True, supervision=False),
        "Demi cycles + Supervision": cond_losses(demi_cycle=True, cycle=False, supervision=True),
        "Complete cycles + Supervision": cond_losses(demi_cycle=False, cycle=True, supervision=True),
        "All losses": cond_losses(demi_cycle=True, cycle=True, supervision=True),
    }

    alphas = {
        "Supervision": 1,
        "Demi cycles + Supervision": 1,
        "Complete cycles + Supervision": 1,
        "All losses": 1,

    }

    for used_loss, loss_label in losses.items():
        for k, (label, cond) in enumerate(loss_parts.items()):
            l = log.filter_neq(f'metrics/val_in_dist_supervision_{used_loss} (last)', '')
            l = l.filter(cond)
            if label != "Supervision":
                l = l.filter_eq("parameters/losses/coefs/supervision", alpha)
            lf = l.filter_between('metrics/epoch (last)', min_epoch)
            x, y, ids = lf.zip(['n_images', f'metrics/val_in_dist_supervision_{used_loss} (last)', '_token'], sort=lambda e: e[0])
            line = plt.loglog(x, y, "x-", alpha=alphas[label], label=label)

            # lf = l.filter_between('metrics/epoch (last)', 90)
            # x, y = lf.zip(['n_images', f'metrics/val_loss_supervision_{used_loss} (last)'], sort=lambda e: e[0])
            # if k == len(loss_parts.keys()) - 1:
            #     plt.loglog(x, y, "x", c=line[0].get_color(), label="Trained with at least 80 epochs.")
            # else:
            #     plt.loglog(x, y, "x", c=line[0].get_color())

        plt.title("Effect of losses")
        plt.xlabel("Number of synchronized examples")
        plt.ylabel(f"Averaged translation loss on the validation set")
        legend = plt.legend()
        for t in legend.get_texts():
            t.set_alpha(alphas[t.get_text()])
        plt.show()
