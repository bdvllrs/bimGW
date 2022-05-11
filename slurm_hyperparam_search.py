import subprocess
from itertools import product

from omegaconf import OmegaConf, DictConfig


def walk_dict(d, prefix=[]):
    new_dic = {}
    for key, val in d.items():
        if type(val) is DictConfig:
            new_dic.update(walk_dict(val, prefix + [key]))
        else:
            new_dic[".".join(prefix + [key])] = val
    return new_dic


if __name__ == '__main__':
    args = OmegaConf.from_cli()
    print(args)
    unstructured_dict = walk_dict(args)
    print(unstructured_dict)

    keys, values = zip(*unstructured_dict.items())
    all_combinations = list(product(*values))

    for combination in all_combinations:
        name = "_".join([f"{p.split('.')[-1]}_{v}" for p, v in zip(keys, combination)])
        command = ["bash", "submitSLURMScript.sh", f"bim_gw_{name}", "train.py", "1", "12:00:00", "1", "90000",
                   "benjam.devillers@gmail.com"]
        for name, val in zip(keys, combination):
            command.append(f"{name}={val}")
        # print(command)
        subprocess.run(command)
