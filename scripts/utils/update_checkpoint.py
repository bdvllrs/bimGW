import torch

state_dict_rename_to = {
    "domain_mods": "domains._domain_modules",
    "decoders.v.encoder_head.0": "decoders.v.encoder_head.z_img",
    "decoders.t.encoder_head.0": "decoders.t.encoder_head.z",
    # "_domain_modules.v": "_domain_modules.v.workspace_module",
}


def import_v(path: str):
    model = torch.load(path)
    new_state_dict = {}
    before_keys = list(model["state_dict"].keys())
    for name, val in model["state_dict"].items():
        new_name = name
        for old, new in state_dict_rename_to.items():
            new_name = new_name.replace(old, new)
        new_state_dict[new_name] = val
    model["state_dict"] = new_state_dict
    after_keys = list(model["state_dict"].keys())
    for before_key, after_key in zip(before_keys, after_keys):
        print(f"B {before_key}")
        print(f"A {after_key}")

    torch.save(model, path)


if __name__ == "__main__":
    import_v(
        "/home/benjamin/Projects/bimGW/checkpoints/gw_t-v/gw_v-t_N=500000_tr+dcy+cy_1058978.ckpt"
    )
