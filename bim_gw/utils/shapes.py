
import cv2
import numpy as np
from matplotlib import patches as patches, pyplot as plt, gridspec
from neptune.new.types import File


def get_square_patch(location, radius, rotation, color):
    x, y = location[0], location[1]
    coordinates = np.array([[-radius, radius],
                            [radius, radius],
                            [radius, -radius],
                            [-radius, -radius]])
    origin = np.array([[x, y], [x, y], [x, y], [x, y]])
    rotation_m = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    patch = patches.Polygon(origin + coordinates @ rotation_m, facecolor=color)
    return patch


def get_triangle_patch(location, radius, rotation, color):
    x, y = location[0], location[1]
    coordinates = np.array([[-radius, 0],
                            [radius, 0],
                            [radius, -radius]])
    origin = np.array([[x, y], [x, y], [x, y]])
    rotation_m = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    patch = patches.Polygon(origin + coordinates @ rotation_m, facecolor=color)
    return patch


def get_circle_patch(location, radius, rotation, color):
    x, y = location[0], location[1]
    patch = patches.Circle((x, y), radius, facecolor=color)
    return patch


def generate_image(ax, cls, location, radius, rotation, color, imsize=32):
    if cls == 0:
        patch = get_square_patch(location, radius, rotation, color)
    elif cls == 1:
        patch = get_circle_patch(location, radius, rotation, color)
    elif cls == 2:
        patch = get_triangle_patch(location, radius, rotation, color)
    else:
        raise ValueError("Class does not exist.")

    ax.add_patch(patch)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    # ax.axis('off')
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)


def get_fig_from_specs(cls, locations, radii, rotations, colors, imsize=32, ncols=8):
    dpi = 100.
    nrows = len(cls) // ncols

    width = ncols * (imsize + 1) + 1
    height = nrows * (imsize + 1) + 1

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    gs = gridspec.GridSpec(
        nrows, ncols,
        wspace=0,
        hspace=0,
        # hspace=1 / (imsize + 3),
        left=0,
        # right=(1 - 1 / imsize),
        right=1,
        bottom=0,
        top=1
        # bottom=1 / imsize,
        # top=(1 - 1 / imsize)
    )
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j
            ax = plt.subplot(gs[i, j])
            generate_image(ax, cls[k], locations[k], radii[k], rotations[k], colors[k], imsize)
    return fig


def get_image_specs_from_latents(cls, latents):
    output = dict()
    output['cls'] = cls
    output['locations'] = np.stack((latents[:, 0], latents[:, 1]), axis=1)
    output['radii'] = latents[:, 2]
    output['rotations'] = latents[:, 3]
    output['colors'] = np.stack((latents[:, 4], latents[:, 5], latents[:, 6]), axis=1)
    return output


def log_shape_fig(logger, classes, latents, name):
    spec = get_image_specs_from_latents(classes, latents)
    fig = get_fig_from_specs(**spec)
    # plt.tight_layout(pad=0)

    if logger is not None:
        logger.experiment[name].log(File.as_image(fig))
    else:
        plt.show()
    plt.close(fig)


def generate_radius(n_samples, min, max):
    assert max > min
    return np.random.randint(min, max, n_samples)


def generate_color(n_samples, max_lightness=256):
    assert 0 <= max_lightness <= 256
    hls = np.random.randint([0, 0, 0], [181, max_lightness, 256], size=(1, n_samples, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0].astype(np.float) / 255
    return rgb


def generate_rotation(n_samples):
    return np.random.rand(n_samples) * 360


def generate_location(n_samples, radius, imsize):
    assert (radius <= imsize / (2 * np.sqrt(2))).all()
    radii = np.sqrt(2) * np.stack((radius, radius), axis=1)
    locations = np.random.randint(radii, imsize - radii, (n_samples, 2))
    return locations


def generate_class(n_samples, classes):
    return np.random.randint(len(classes), size=n_samples)


def generate_dataset(n_samples, class_names, min_radius, max_radius, max_lightness, imsize):
    classes = generate_class(n_samples, class_names)
    sizes = generate_radius(n_samples, min_radius, max_radius)
    locations = generate_location(n_samples, sizes, imsize)
    rotation = generate_rotation(n_samples)
    colors = generate_color(n_samples, max_lightness)
    return dict(classes=classes, locations=locations, sizes=sizes, rotations=rotation, colors=colors)