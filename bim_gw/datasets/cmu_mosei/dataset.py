import torch
from mmsdk import mmdatasdk


class CMUMOSEIDataset:
    def __init__(self, cmu_dataset, split, selected_domains, transforms=None):
        self.split = split
        self.selected_domains = selected_domains
        self.transforms = transforms

        self.dataset = cmu_dataset

    def __len__(self):
        return self.dataset["All Labels"].shape[0]

    def __getitem__(self, item):
        data = {
            key: torch.from_numpy(self.dataset[name][item])
            for key, name in self.selected_domains.items()
        }
        data['s'] = self.dataset["All Labels"][item]
        if self.transforms is not None:
            data = self.transforms(data)
        return data
