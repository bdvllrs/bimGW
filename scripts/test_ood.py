import os

import matplotlib.pyplot as plt
import torch.nn.functional
import torchvision
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.datasets.utils import get_lm
from bim_gw.modules.gw import GlobalWorkspace
from bim_gw.modules.vae import VAE
from bim_gw.utils import get_args
from bim_gw.utils.shapes import log_shape_fig


def test_ood(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace, bimodal=True)
    data.prepare_data()
    data.setup(stage="fit")

    vae = VAE.load_from_checkpoint(
        args.global_workspace.vae_checkpoint,
        mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
        kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
    ).eval()
    vae.freeze()

    lm = get_lm(args, data)
    lm.freeze()

    global_workspace = GlobalWorkspace.load_from_checkpoint(args.checkpoint, domain_mods={
        "v": vae,
        "t": lm
    })
    global_workspace.eval()

    print("OOD t --> v")
    possible_2_class = torch.tensor([
        [0.5, 0., 0.],
        [0., 0.5, 0.],
        [0., 0., 0.5],
        [1., 1., 0.],
        [1., 0., 1.],
        [0., 1., 1.],
        [0.5, 0.5, 0.],
        [0.5, 0., 0.5],
        [0., 0.5, 0.5],
        [0.33, 0.33, 0.33],
        [0., 0., 0.],
    ])
    t_samples = lm.sample(1)
    t_latent = lm.encode(t_samples)
    class_labels =  possible_2_class
    print(class_labels)
    t_latent = [class_labels, t_latent[1].expand(class_labels.size(0), -1)]
    w = global_workspace.encode(t_latent, "t")
    t_demi_cycle_latent = global_workspace.decode(w, "t")
    t_demi_cycle = lm.decode(t_demi_cycle_latent)
    v_translation = global_workspace.decode(w, "v")
    t_recons_latent = global_workspace.translate(v_translation, "v", "t")
    t_recons = lm.decode(t_recons_latent)
    v_images = vae.decode(v_translation)

    render_semantic(t_demi_cycle, "Rendered after demi-cycle (t → GW → t)")
    render_semantic(t_recons, "Rendered after full-cycle (t → v → t)")
    plot_image_grid(v_images, "Reconstruction after translation (t → v)")

    print("OOD v --> t")
    v_latent = vae.encode(data.validation_domain_examples["v"])
    # v_latent = torch.randn(32, vae.z_size)
    # v_latent = torch.zeros(32, vae.z_size)
    v_image = vae.decoder(v_latent)
    w = global_workspace.encode(v_latent, "v")
    print("Diff GW and latent", torch.nn.functional.mse_loss(v_latent, w))
    v_demi_cycle = global_workspace.decode(w, "v")
    t_latent = global_workspace.translate(v_latent, "v", "t")
    v_back_translation = global_workspace.translate(t_latent, "t", "v")
    t_recons = lm.decode(t_latent)
    v_recons = vae.decode(v_back_translation)
    v_recons_demi_cycle = vae.decode(v_demi_cycle)

    plot_image_grid(data.validation_domain_examples["v"], "Original images")
    plot_image_grid(v_image, "VAE reconstruction")
    plot_image_grid(v_recons_demi_cycle, "Reconstruction demi-cycle (v → gw → v)")
    plot_image_grid(v_recons, "Reconstruction full cycle (v → t → v)")
    render_semantic(t_recons, "Rendered images after translation (v → t)")

    print("Look at global workspace")

    latent_t = global_workspace.domain_mods["t"].encode(data.validation_domain_examples["t"])
    z_t = global_workspace.encoders["t"](latent_t)

    latent_v = global_workspace.domain_mods["v"].encode(data.validation_domain_examples["v"])
    z_v = global_workspace.encoders["v"](latent_v)

    print(torch.nn.functional.mse_loss(z_t, z_v))


def plot_image_grid(images, title):
    img_grid = torchvision.utils.make_grid(images, normalize=True)
    img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
    plt.imshow(img_grid)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.tight_layout(pad=0)
    plt.title(title)
    plt.show()

def render_semantic(t_domain, title):
    classes = t_domain[0][:32].detach().cpu().numpy()
    latents = t_domain[1][:32].detach().cpu().numpy()
    log_shape_fig(None, classes, latents, title)

if __name__ == "__main__":
    test_ood(get_args(debug=int(os.getenv("DEBUG", 0))))
