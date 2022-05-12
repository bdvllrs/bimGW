import os
from pathlib import Path
from auto_sbatch import auto_sbatch
from omegaconf import OmegaConf

from bim_gw.utils import get_args

if __name__ == '__main__':
    work_dir = Path(__file__).parent.parent
    script_location = work_dir / "scripts/train.py"

    args = get_args(debug=int(os.getenv("DEBUG", 0))).slurm
    args = OmegaConf.merge(args, {
        "work_directory": str(work_dir.absolute()),
        "script_location": str(script_location.absolute())
    })

    auto_sbatch(args)

