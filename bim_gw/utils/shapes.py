import io

import numpy as np
from matplotlib import patches as patches, pyplot as plt, gridspec


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
    ax.axis('off')
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)


def get_shape_array(cls, locations, radii, rotations, colors, imsize=32, ncols=8):
    nrows = len(cls) // ncols
    fig = plt.figure(figsize=(nrows * imsize, ncols * imsize), dpi=1)

    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1, 1, 1],
                           wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
    for i in range(nrows):
        for j in range(ncols):
            k = i * nrows + j
            ax = plt.subplot(gs[i, j])
            generate_image(ax, cls[k], locations[k], radii[k], rotations[k], colors[k], imsize)

    return fig


def get_image_specs_from_latents(cls, latents):
    output = dict()
    output['class'] = cls.argmax(-1)
    output['x'] = latents[:, 0]
    output['y'] = latents[:, 1]
    output['radii'] = latents[:, 2]
    output['rotations'] = latents[:, 3]
    output['r'] = latents[:, 4]
    output['g'] = latents[:, 5]
    output['b'] = latents[:, 6]
    return output