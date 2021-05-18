<div align="center">
    <h1>Bim GW<br><small>Bimodal Global Workspace</small></h1>
</div>

# Setup
## For development
```
git clone git@github.com:bdvllrs/bimGW.git & cd bimGW
pip install -e .
```

## Quick install
```
pip install git+https://github.com/bdvllrs/bimGW.git
```

## Configurations
They can be found in `config/`.

It contains `config/main.yaml` which is the source configuration file. To use different values, 
create a new `config/local.yaml` with updated values.

You can also create a `config/debug.yaml` which will only be loaded if the `DEBUG` environment
variable is set to `1`.

## Datasets
This project requires:
- ImageNet: its path is to be filled in `config/local.yaml` if the field `image_net_path`.

# Run
The scripts are located in the `scripts` folder.
- Run `train_vae.py` to train the vision model.
