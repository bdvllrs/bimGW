import matplotlib.path as mpath
import numpy as np
from matplotlib import patches as patches, pyplot as plt, gridspec
from neptune.new.types import File


def get_transformed_coordinates(coordinates, origin, scale, rotation):
    center = np.array([[0.5, 0.5]])
    rotation_m = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    rotated_coordinates = (coordinates - center) @ rotation_m.T
    return origin + scale * rotated_coordinates


def get_diamond_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    coordinates = np.array([[0.5, 0],
                            [1, 0.3],
                            [0.5, 1],
                            [0, 0.3]])
    origin = np.array([[x, y]])
    patch = patches.Polygon(get_transformed_coordinates(coordinates, origin, scale, rotation), facecolor=color)
    return patch


def get_square_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    shift = (2 - np.sqrt(2)) / 4
    coordinates = np.array([[shift, shift],
                            [1 - shift, shift],
                            [1 - shift, 1 - shift],
                            [shift, 1 - shift]])
    patch = patches.Polygon(get_transformed_coordinates(coordinates, origin, scale, rotation), facecolor=color)
    return patch


def get_triangle_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array([[0.5, 1],
                            [0.2, 0],
                            [0.8, 0]])
    patch = patches.Polygon(get_transformed_coordinates(coordinates, origin, scale, rotation), facecolor=color)
    return patch


def get_circle_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    patch = patches.Circle((x, y), scale / 2, facecolor=color)
    return patch


def get_egg_patch(location, scale, rotation, color):
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array([[0.5, 0],
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
                            [0.5, 0]])
    codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
             mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
             mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
    path = mpath.Path(get_transformed_coordinates(coordinates, origin, scale, rotation), codes)
    patch = patches.PathPatch(path, facecolor=color)
    return patch


def generate_image(ax, cls, location, scale, rotation, color, imsize=32):
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
            ax.set_facecolor("black")
    return fig


def get_image_specs_from_latents(cls, latents):
    output = dict()
    output['cls'] = cls
    output['locations'] = np.stack((latents[:, 0], latents[:, 1]), axis=1)
    output['radii'] = latents[:, 2]
    output['rotations'] = latents[:, 3] * 2 * np.pi
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


def generate_scale(n_samples, min_val, max_val):
    assert max_val > min_val
    return np.random.randint(min_val, max_val, n_samples)
    # return np.full(n_samples, 32)


def generate_color(n_samples, min_lightness=0, max_lightness=256):
    import cv2

    assert 0 <= max_lightness <= 256
    hls = np.random.randint([0, min_lightness, 0], [181, max_lightness, 256], size=(1, n_samples, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0].astype(np.float) / 255
    return rgb


def generate_rotation(n_samples, classes):
    rotations = np.random.rand(n_samples) * 2 * np.pi
    # rotations[classes == 1] = 0  # circles don't have rotations
    # rotations[classes == 0] = rotations[classes == 0] % 90
    # return np.zeros(n_samples)
    return rotations


def generate_location(n_samples, scale, imsize):
    assert (scale <= imsize).all()
    radii = 1.2 * np.stack((scale / 2, scale / 2), axis=1)
    locations = np.random.randint(radii, imsize - radii, (n_samples, 2))
    # locations = np.full((n_samples, 2), imsize // 2)
    return locations


def generate_class(n_samples):
    return np.random.randint(3, size=n_samples)


def generate_dataset(n_samples, min_scale, max_scale, min_lightness, max_lightness, imsize,
                     classes=None):
    if classes is None:
        classes = generate_class(n_samples)
    sizes = generate_scale(n_samples, min_scale, max_scale)
    locations = generate_location(n_samples, sizes, imsize)
    rotation = generate_rotation(n_samples, classes)
    colors = generate_color(n_samples, min_lightness, max_lightness)
    return dict(classes=classes, locations=locations, sizes=sizes, rotations=rotation, colors=colors)
