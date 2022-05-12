import subprocess
from itertools import product

from omegaconf import OmegaConf, DictConfig, ListConfig


def walk_dict(d, prefix=[]):
    new_dic = {}
    for key, val in d.items():
        if type(val) is DictConfig:
            new_dic.update(walk_dict(val, prefix + [key]))
        else:
            if isinstance(val, ListConfig):
                new_dic[".".join(prefix + [key])] = val
    return new_dic


if __name__ == '__main__':
    args = OmegaConf.from_cli()
    print(args)
    unstructured_dict = walk_dict(args)
    print(unstructured_dict)

    keys, values = zip(*unstructured_dict.items())
    all_combinations = list(product(*values))

    new_dict = [
        f"{key}=[{', '.join([str(all_combinations[i][k]) for i in range(len(all_combinations))])}]" for k, key in enumerate(keys)
    ]
    new_values = OmegaConf.from_dotlist(new_dict)

    for combination in all_combinations:
        name = "_".join([f"{p.split('.')[-1]}_{v}" for p, v in zip(keys, combination)])
        command = ["bash", "submitSLURMScript.sh", f"bim_gw_{name}", "train.py", "1", "12:00:00", "1", "90000",
                   "benjam.devillers@gmail.com"]
        for name, val in zip(keys, combination):
            command.append(f"{name}={val}")
        print(command)
        # subprocess.run(command)
