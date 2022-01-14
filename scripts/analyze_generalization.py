import matplotlib.pyplot as plt
import numpy as np

from bim_gw.utils.neptune import CSVLog


if __name__ == '__main__':
    log = CSVLog("../data/bim-gw.csv")
    log = log.add_token_column().filter_between("_token", 861)
    log = log.add_column("n_images", lambda row: int(row["prop_lab_images"] * 500_000))

    alpha = 4
    min_epoch = 1
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    prop_lab_images = [0.002, 0.01]

    log = log.filter_in("parameters/seed", seeds).filter_between('metrics/epoch (last)', min_epoch)
    for prop_image in prop_lab_images:
        l = log.filter_eq("prop_lab_images", prop_image)
        avg_acc_id = np.mean([d['metrics/val_in_dist_supervision_loss (last)'] for d in l.data])
        std_acc_id = np.var([d['metrics/val_in_dist_supervision_loss (last)'] for d in l.data])
        avg_acc_ood = np.mean([d['metrics/val_ood_supervision_loss (last)'] for d in l.data])
        std_acc_ood = np.var([d['metrics/val_ood_supervision_loss (last)'] for d in l.data])
        print(f"Prop sync: {prop_image}")
        print(f"ID: {avg_acc_id} +/- {std_acc_id}")
        print(f"OOD: {avg_acc_ood} +/- {std_acc_ood}")