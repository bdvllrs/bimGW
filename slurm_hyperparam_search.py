from itertools import product

from omegaconf import OmegaConf

if __name__ == '__main__':
    args = OmegaConf.from_cli()
    print(args)

    keys, values = zip(*args.items())
    all_combinations = list(product(*values))

    for combination in all_combinations:
        name = "_".join([f"{p}_{v}" for p, v in zip(keys, combination)])
        command = ["bash", "submitSLURMScript.sh", f"bim_gw_{name}", "train", "1", "12:00:00", "1", "90000",
                   "benjam.devillers@gmail.com"]
        for name, val in zip(keys, combination):
            command.append(f"{name}={val}")
        print(command)
        # subprocess.run(command)
