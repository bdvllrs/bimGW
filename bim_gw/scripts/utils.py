from bim_gw.modules import VAE, ShapesLM, ActionModule
from bim_gw.modules.language_model import ShapesAttributesLM
from bim_gw.modules.workspace_module import PassThroughWM


def get_domain(name, domain_name, args, data):
    if domain_name in ["v", "v_f"]:
        domain = VAE.load_from_checkpoint(
            args.global_workspace.vae_checkpoint,
            mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
            kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
        )
    elif domain_name in ["t", "t_f"]:
        domain = ShapesLM.load_from_checkpoint(
            args.global_workspace.lm_checkpoint,
            bert_path=args.global_workspace.bert_path)
    elif domain_name in ["attr", "attr_f"]:
        domain = ShapesAttributesLM(len(data.classes), data.img_size)
    elif domain_name == "a":
        domain = ActionModule(len(data.classes), data.img_size)
    else:
        raise ValueError(f"{domain_name} is not a valid domain name.")

    if args.global_workspace.use_pre_saved and name in args.global_workspace.load_pre_saved_latents.keys():
        domain = PassThroughWM(domain)

    domain.eval()
    domain.freeze()
    return domain


def get_domains(args, data):
    return {
        name: get_domain(name, domain, args, data) for name, domain in args.global_workspace.selected_domains.items()
    }