import matplotlib.path as mpath
import numpy as np
from matplotlib import gridspec
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_transformed_coordinates(coordinates, origin, scale, rotation):
    center = np.array([[0.5, 0.5]])
    rotation_m = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    rotated_coordinates = (coordinates - center) @ rotation_m.T
    return origin + scale * rotated_coordinates


def get_diamond_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    coordinates = np.array([[0.5, 0], [1, 0.3], [0.5, 1], [0, 0.3]])
    origin = np.array([[x, y]])
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def get_square_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    shift = (2 - np.sqrt(2)) / 4
    coordinates = np.array(
        [
            [shift, shift],
            [1 - shift, shift],
            [1 - shift, 1 - shift],
            [shift, 1 - shift],
        ]
    )
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def get_triangle_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array([[0.5, 1], [0.2, 0], [0.8, 0]])
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def get_circle_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    patch = patches.Circle((x, y), scale / 2, facecolor=color)
    return patch


def get_egg_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array(
        [
            [0.5, 0],
            [0.8, 0],
            [0.9, 0.1],
            [0.9, 0.3],
            [0.9, 0.5],
            [0.7, 1],
            [0.5, 1],
            [0.3, 1],
            [0.1, 0.5],
            [0.1, 0.3],
            [0.1, 0.1],
            [0.2, 0],
            [0.5, 0],
        ]
    )
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
    ]
    path = mpath.Path(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        codes,
    )
    patch = patches.PathPatch(path, facecolor=color)
    return patch


def generate_image(ax, cls, location, scale, rotation, color, imsize=32):
    color = color.astype(np.float32) / 255
    if cls == 0:
        patch = get_diamond_patch(location, scale, rotation, color)
    elif cls == 1:
        patch = get_egg_patch(location, scale, rotation, color)
    elif cls == 2:
        patch = get_triangle_patch(location, scale, rotation, color)
    else:
        raise ValueError("Class does not exist.")

    ax.add_patch(patch)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    # ax.axis('off')
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)


def get_fig_from_specs(
    cls, locations, radii, rotations, colors, imsize=32, ncols=8
):
    dpi = 100.0
    reminder = 1 if len(cls) % ncols else 0
    nrows = len(cls) // ncols + reminder

    width = ncols * (imsize + 1) + 1
    height = nrows * (imsize + 1) + 1

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    gs = gridspec.GridSpec(
        nrows,
        ncols,
        wspace=1 / 10,
        hspace=1 / 10,
        left=0,
        right=1,
        bottom=0,
        top=1,
    )
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j
            if k >= len(cls):
                break
            ax = plt.subplot(gs[i, j])
            generate_image(
                ax,
                cls[k],
                locations[k],
                radii[k],
                rotations[k],
                colors[k],
                imsize,
            )
            ax.set_facecolor("black")
    return fig


def get_image_specs_from_latents(cls, latents):
    output = dict()
    output["cls"] = cls
    output["locations"] = np.stack((latents[:, 0], latents[:, 1]), axis=1)
    output["radii"] = latents[:, 2]
    rotation_x = latents[:, 3] * 2 - 1
    rotation_y = latents[:, 4] * 2 - 1
    output["rotations"] = np.arctan2(rotation_y, rotation_x)
    output["colors"] = (
        np.stack((latents[:, 5], latents[:, 6], latents[:, 7]), axis=1) * 255
    )
    return output


def log_shape_fig(logger, classes, latents, name, step=None):
    spec = get_image_specs_from_latents(classes, latents)
    fig = get_fig_from_specs(**spec)

    if logger is not None:
        logger.log_image(name, fig, step=step)
    else:
        plt.title(name)
        plt.show()
    plt.close(fig)


def generate_scale(n_samples, min_val, max_val):
    assert max_val > min_val
    return np.random.randint(min_val, max_val + 1, n_samples)


def generate_color(n_samples, min_lightness=0, max_lightness=256):
    import cv2

    assert 0 <= max_lightness <= 256
    hls = np.random.randint(
        [0, min_lightness, 0],
        [181, max_lightness, 256],
        size=(1, n_samples, 3),
        dtype=np.uint8,
    )
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]
    return rgb.astype(int), hls[0].astype(int)


def generate_rotation(n_samples):
    rotations = np.random.rand(n_samples) * 2 * np.pi
    return rotations


def generate_location(n_samples, max_scale, imsize):
    assert max_scale <= imsize
    margin = max_scale / 2
    locations = np.random.randint(margin, imsize - margin, (n_samples, 2))
    return locations


def generate_class(n_samples):
    return np.random.randint(3, size=n_samples)


def generate_unpaired_attr(n_samples):
    return np.random.rand(n_samples)


def generate_transformations(labels, target_labels):
    n_samples = len(labels["classes"])
    assert len(target_labels["classes"]) == n_samples
    transformation_complexity = np.random.rand(n_samples)
    transformations = np.zeros((n_samples, 12))
    for k in range(n_samples):
        n_transfo = 1
        if 0.6 <= transformation_complexity[k] <= 0.9:
            n_transfo = 2
        elif transformation_complexity[k] > 0.9:
            n_transfo = 3
        for i in range(n_transfo):
            transfo = np.random.randint(0, 6)
            transformations[k, 0] = labels["classes"][k]
            if transfo == 0:  # class
                transformations[k, 0] = target_labels["classes"][k]
            elif transfo == 1:  # scale
                transformations[k, 1] = (
                    target_labels["sizes"][k] - labels["sizes"][k]
                )
            elif transfo == 2:  # location
                transformations[k, 2] = (
                    target_labels["locations"][k, 0]
                    - labels["locations"][k, 0]
                )
                transformations[k, 3] = (
                    target_labels["locations"][k, 1]
                    - labels["locations"][k, 1]
                )
            elif transfo == 3:  # rotation
                transformations[k, 4] = (
                    target_labels["rotations"][k] - labels["rotations"][k]
                )
            elif transfo == 4:  # color
                transformations[k, 5] = (
                    target_labels["colors"][k, 0] - labels["colors"][k, 0]
                )
                transformations[k, 6] = (
                    target_labels["colors"][k, 1] - labels["colors"][k, 1]
                )
                transformations[k, 7] = (
                    target_labels["colors"][k, 2] - labels["colors"][k, 2]
                )
                transformations[k, 8] = (
                    target_labels["colors_hls"][k, 0]
                    - labels["colors_hls"][k, 0]
                )
                transformations[k, 9] = (
                    target_labels["colors_hls"][k, 1]
                    - labels["colors_hls"][k, 1]
                )
                transformations[k, 10] = (
                    target_labels["colors_hls"][k, 2]
                    - labels["colors_hls"][k, 2]
                )
            else:
                transformations[k, 11] = 0
    return (
        transformed_labels(labels, transformations),
        labels_from_transfo(transformations),
    )


def transformed_labels(labels, transfo):
    return dict(
        classes=transfo[:, 0],
        locations=labels["locations"] + transfo[:, 2:4],
        sizes=labels["sizes"] + transfo[:, 1],
        rotations=labels["rotations"] + transfo[:, 4],
        colors=labels["colors"] + transfo[:, 5:8],
        colors_hls=labels["colors_hls"] + transfo[:, 8:11],
        unpaired=labels["unpaired"] + transfo[:, 11],
    )


def labels_from_transfo(transfo):
    return dict(
        classes=transfo[:, 0],
        locations=transfo[:, 2:4],
        sizes=transfo[:, 1],
        rotations=transfo[:, 4],
        colors=transfo[:, 5:8],
        colors_hls=transfo[:, 8:11],
        unpaired=transfo[:, 11],
    )


def generate_dataset(
    n_samples,
    min_scale,
    max_scale,
    min_lightness,
    max_lightness,
    imsize,
    classes=None,
):
    if classes is None:
        classes = generate_class(n_samples)
    sizes = generate_scale(n_samples, min_scale, max_scale)
    locations = generate_location(n_samples, max_scale, imsize)
    rotation = generate_rotation(n_samples)
    colors_rgb, colors_hls = generate_color(
        n_samples, min_lightness, max_lightness
    )
    unpaired = generate_unpaired_attr(n_samples)
    return dict(
        classes=classes,
        locations=locations,
        sizes=sizes,
        rotations=rotation,
        colors=colors_rgb,
        colors_hls=colors_hls,
        unpaired=unpaired,
    )


def save_dataset(path_root, dataset, imsize):
    dpi = 1
    classes, locations = dataset["classes"], dataset["locations"]
    radii = dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    enumerator = tqdm(
        enumerate(zip(classes, locations, radii, rotations, colors)),
        total=len(classes),
    )
    for k, (cls, location, scale, rotation, color) in enumerator:
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        generate_image(ax, cls, location, scale, rotation, color, imsize)
        ax.set_facecolor("black")
        plt.tight_layout(pad=0)
        plt.savefig(path_file, dpi=dpi, format="png")
        plt.close(fig)


def load_labels_old(path_root):
    labels = np.load(str(path_root))
    dataset = {
        "classes": labels[:, 0],
        "locations": labels[:, 1:3],
        "sizes": labels[:, 3],
        "rotations": labels[:, 4],
        "colors": labels[:, 5:8],
        "colors_hls": labels[:, 8:11],
    }
    dataset_transfo = {
        "classes": labels[:, 11],
        "locations": labels[:, 12:14],
        "sizes": labels[:, 14],
        "rotations": labels[:, 15],
        "colors": labels[:, 16:19],
        "colors_hls": labels[:, 19:22],
    }
    return dataset, dataset_transfo


def load_labels(path_root):
    labels = np.load(str(path_root))
    dataset = {
        "classes": labels[:, 0],
        "locations": labels[:, 1:3],
        "sizes": labels[:, 3],
        "rotations": labels[:, 4],
        "colors": labels[:, 5:8],
        "colors_hls": labels[:, 8:11],
        "unpaired": labels[:, 11],
    }
    dataset_transfo = {
        "classes": labels[:, 11],
        "locations": labels[:, 12:14],
        "sizes": labels[:, 14],
        "rotations": labels[:, 15],
        "colors": labels[:, 16:19],
        "colors_hls": labels[:, 19:22],
        "unpaired": labels[:, 22],
    }
    return dataset, dataset_transfo


def save_labels(path_root, dataset, dataset_transfo):
    classes, locations = dataset["classes"], dataset["locations"]
    sizes = dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    colors_hls = dataset["colors_hls"]
    unpaired_attr = dataset["unpaired"]
    classes_transfo = dataset_transfo["classes"]
    locations_transfo = dataset_transfo["locations"]

    sizes_transfo = dataset_transfo["sizes"]
    rotations_transfo = dataset_transfo["rotations"]

    colors_transfo = dataset_transfo["colors"]
    colors_hls_transfo = dataset_transfo["colors_hls"]
    unpaired_attr_transfo = dataset_transfo["unpaired"]
    # TODO: add transformation to unpaired examples
    labels = np.concatenate(
        [
            classes.reshape((-1, 1)),
            locations,
            sizes.reshape((-1, 1)),
            rotations.reshape((-1, 1)),
            colors,
            colors_hls,
            unpaired_attr.reshape((-1, 1)),
            classes_transfo.reshape((-1, 1)),
            locations_transfo,
            sizes_transfo.reshape((-1, 1)),
            rotations_transfo.reshape((-1, 1)),
            colors_transfo,
            colors_hls_transfo,
            unpaired_attr_transfo.reshape((-1, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    np.save(path_root, labels)
