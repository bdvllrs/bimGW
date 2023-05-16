from typing import Any, Callable

from torchvision import transforms


class ComposeWithExtraParameters:
    """
    DomainLoaders return DomainItems we apply the transform only
    on the modality
    """

    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, x):
        for key, transform in self.transforms.items():
            x[key] = transform(x[key])
        return x


def get_v_preprocess(augmentation: bool = False) -> Callable[[Any], Any]:
    transformations = []
    if augmentation:
        transformations.append(transforms.RandomHorizontalFlip())

    transformations.extend(
        [
            transforms.ToTensor(),
            # transforms.Normalize(norm_mean, norm_std)
        ]
    )

    return ComposeWithExtraParameters(
        {"img": transforms.Compose(transformations)}
    )
