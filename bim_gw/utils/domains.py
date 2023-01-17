from bim_gw.modules import VAE
from bim_gw.modules.language_model import ShapesAttributesLM, ShapesLM


class DomainRegistry:
    __instance = None
    _domain_callbacks = {}

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance

    def add(self, key, domain):
        if key not in self._domain_callbacks:
            self._domain_callbacks[key] = domain
        return self

    def get(self, key):
        return self._domain_callbacks[key]


def add_domains_to_registry():
    domain_registry = DomainRegistry()
    domain_registry.add("v", lambda args, img_size=None: VAE.load_from_checkpoint(
        args.global_workspace.vae_checkpoint,
        mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
        kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
        strict=False
    ))
    domain_registry.add("attr", lambda args, img_size: ShapesAttributesLM(img_size))
    domain_registry.add("t", lambda args, img_size=None: ShapesLM.load_from_checkpoint(
        args.global_workspace.lm_checkpoint,
        bert_path=args.global_workspace.bert_path,
        z_size=args.lm.z_size,
        hidden_size=args.lm.hidden_size
    ))
