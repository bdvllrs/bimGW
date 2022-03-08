import os

from bim_gw.scripts.train import train_lm
from bim_gw.utils import get_args

if __name__ == "__main__":
    # with torch.profiler.profile(with_stack=True) as prof:
    train_lm(get_args(debug=int(os.getenv("DEBUG", 0))))
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
