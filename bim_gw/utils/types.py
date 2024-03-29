import pathlib
from collections.abc import Container
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Dict, List, Literal, Optional, Protocol, Sequence,
                    TypeVar, Union)

import numpy as np
from numpy.typing import ArrayLike
from omegaconf import MISSING, ListConfig

WandbFilterT = Dict[str, Any]

SplitLiteral = Literal["train", "val", "test"]
ListType = Union[List, ListConfig, ArrayLike]

SequenceLike = Union[np.ndarray, Sequence]

T_co = TypeVar("T_co", covariant=True)
DistLiteral = Literal["in_dist", "ood"]


class SubsetableDataset(Protocol[T_co]):
    def subset(self, indices: Sequence[int]) -> T_co:
        ...


class AvailableDomains(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name, start: int, count: int, last_values: List[Any]
    ) -> str:
        return str(name).lower()


class VAEType(Enum):
    beta = "beta"
    sigma = "sigma"
    optimal_sigma = "optimal_sigma"


class SchedulerMode(Enum):
    fixed = "fixed"
    adaptive = "adaptive"


class SchedulerInterval(Enum):
    step = "step"
    epoch = "epoch"


class AvailableLoggers(Enum):
    NeptuneLogger = "NeptuneLogger"
    WandbLogger = "WandbLogger"
    CSVLogger = "CSVLogger"
    TensorBoardLogger = "TensorBoardLogger"
    MLFlowLogger = "MLFlowLogger"
    AimLogger = "AimLogger"


class LoadFromData(Enum):
    wandb = "wandb"
    csv = "csv"


class LoadFromPath(Enum):
    remote = "remote"
    local = "local"


@dataclass
class ComplexPathInfo(Container):
    load_from: LoadFromPath = MISSING

    remote_server: Optional[str] = None
    remote_user: Optional[str] = None
    remote_password: Optional[str] = None
    remote_checkpoint_path: Optional[str] = None

    local_path: Optional[str] = None


PathType = Union[str, pathlib.Path, ComplexPathInfo]


@dataclass
class OptimConfig:
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class LRGWOptimConfig:
    encoders: float = MISSING
    decoders: float = MISSING
    supervised_multiplier: float = MISSING
    unsupervised_multiplier: float = MISSING


@dataclass
class GWOptimConfig:
    weight_decay: float = MISSING
    lr: LRGWOptimConfig = field(default_factory=LRGWOptimConfig)
    unsupervised_losses_after_n_epochs: int = 0


@dataclass
class GWSchedulerConfig:
    mode: SchedulerMode = MISSING
    interval: SchedulerInterval = MISSING
    step: int = MISSING
    gamma: float = MISSING


@dataclass
class SchedulerConfig:
    step: int = MISSING
    gamma: float = MISSING


@dataclass
class SlurmConfig:
    script: str = MISSING
    slurm: Dict[str, Any] = MISSING
    pre_modules: Optional[List[str]] = MISSING
    run_modules: Optional[List[str]] = MISSING
    python_environment: Optional[str] = MISSING
    command: Optional[str] = MISSING
    run_work_directory: Optional[str] = MISSING
    grid_search: Optional[List[str]] = MISSING
    grid_search_exclude: Optional[List[Dict[str, Any]]] = MISSING


@dataclass
class DataloaderConfig:
    num_workers: int = MISSING


@dataclass
class LoggerConfig:
    logger: AvailableLoggers = MISSING
    save_images: bool = True
    save_tables: bool = True
    save_last_images: bool = True
    save_last_tables: bool = True
    watch_model: bool = False
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegendGWResultConfig:
    num_columns: int = MISSING


@dataclass
class AnnotationColFigureConfig:
    y: float = MISSING
    text_yshift: float = 0.0
    loss: str = MISSING
    curve_start: str = MISSING
    curve_end: str = MISSING


@dataclass
class DataSelectorAxesConfig:
    label: str = MISSING
    load_from: LoadFromData = MISSING
    wandb_entity_project: Optional[str] = MISSING
    wandb_filter: Optional[WandbFilterT] = MISSING
    csv_path: Optional[str] = MISSING
    annotations: List[AnnotationColFigureConfig] = field(default_factory=list)


@dataclass
class FigureLossConfig:
    curves: List[str] = MISSING
    label: str = MISSING


@dataclass
class AxesGWResultConfig:
    title: str = MISSING

    transpose_fig: bool = False

    legend_order: List[str] = field(default_factory=list)
    selected_losses: Dict[str, FigureLossConfig] = field(default_factory=dict)
    cols: List[DataSelectorAxesConfig] = field(default_factory=list)


@dataclass
class AdditionalSlugCond:
    slug_value: str = MISSING
    key: str = MISSING
    eq: Any = MISSING


@dataclass
class VisualizationConfig:
    fg_color: str = MISSING
    bg_color: str = MISSING
    font_size: int = MISSING
    font_size_title: int = MISSING
    line_width: int = MISSING
    saved_figure_path: str = MISSING
    total_num_examples: int = MISSING
    argmin_over: str = MISSING
    x_axis: str = MISSING

    hspace: Optional[float] = MISSING
    wspace: Optional[float] = MISSING

    additional_slug_conds: List[AdditionalSlugCond] = field(
        default_factory=list
    )

    loss_definitions: Dict[str, List[str]] = field(default_factory=dict)
    figures: List[AxesGWResultConfig] = field(default_factory=list)
    mix_loss_coefficients: Dict[str, float] = field(default_factory=dict)
    legend: LegendGWResultConfig = field(default_factory=LegendGWResultConfig)


@dataclass
class VAEConfig:
    batch_size: int = MISSING
    beta: float = MISSING
    z_size: int = MISSING
    ae_size: int = MISSING
    type: VAEType = MISSING
    n_fid_samples: int = MISSING
    early_stopping_patience: Optional[int] = MISSING
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LMConfig:
    z_size: int = MISSING
    hidden_size: int = MISSING
    beta: float = MISSING
    train_vae: bool = MISSING
    train_attr_decoders: bool = MISSING
    optimize_vae_with_attr_regression: bool = MISSING
    coef_attr_loss: float = MISSING
    coef_vae_loss: float = MISSING
    batch_size: int = MISSING
    n_validation_examples: int = MISSING
    early_stopping_patience: Optional[int] = MISSING
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class HiddenSizeGlobalWorkspaceConfig:
    encoder: Dict[str, int] = MISSING
    decoder: Dict[str, int] = MISSING


@dataclass
class NLayersGlobalWorkspaceConfig:
    encoder: Dict[str, int] = MISSING
    decoder: Dict[str, int] = MISSING
    decoder_head: Dict[str, int] = MISSING


@dataclass
class GlobalWorkspaceConfig:
    monitor_grad_norms: bool = MISSING
    batch_size: int = MISSING
    selected_domains: List[str] = MISSING
    sync_uses_whole_dataset: bool = MISSING
    use_pre_saved: bool = MISSING
    load_pre_saved_latents: Optional[Dict[str, str]] = MISSING
    split_ood: bool = MISSING
    bert_path: str = MISSING
    z_size: int = MISSING
    prop_labelled_images: float = MISSING
    prop_available_images: float = MISSING
    remove_sync_domains: Optional[List[str]] = MISSING
    vae_checkpoint: Optional[str] = MISSING
    lm_checkpoint: Optional[str] = MISSING
    early_stopping_patience: int = MISSING
    ood_hole_attrs: int = MISSING
    ood_seed: int = MISSING
    ood_idx_domain: int = MISSING
    ood_create_new_examples: bool = MISSING
    ood_folder: Optional[str] = None

    hidden_size: HiddenSizeGlobalWorkspaceConfig = field(
        default_factory=HiddenSizeGlobalWorkspaceConfig
    )
    n_layers: NLayersGlobalWorkspaceConfig = field(
        default_factory=NLayersGlobalWorkspaceConfig
    )
    optim: GWOptimConfig = field(default_factory=GWOptimConfig)
    scheduler: GWSchedulerConfig = field(default_factory=GWSchedulerConfig)


@dataclass
class CoefsLossesConfig:
    demi_cycles: float = MISSING
    cycles: float = MISSING
    translation: float = MISSING
    contrastive: float = MISSING

    # Deprecated
    cosine: float = 0.0
    supervision: Optional[float] = None


@dataclass
class LossesConfig:
    schedules: Optional[Dict[str, SchedulerConfig]] = MISSING
    coefs: CoefsLossesConfig = field(default_factory=CoefsLossesConfig)


@dataclass
class EncoderOddImageConfig:
    selected_id: Optional[str] = MISSING
    use_dist: bool = MISSING
    path: Optional[str] = MISSING
    load_from: Optional[LoadFromData] = MISSING
    wandb_entity_project: Optional[str] = MISSING
    wandb_filter: Optional[WandbFilterT] = MISSING
    csv_path: Optional[str] = MISSING
    selected_id_key: Optional[str] = MISSING


@dataclass
class CheckpointOddImageConfig:
    load_from: Optional[LoadFromData] = MISSING
    wandb_entity_project: Optional[str] = MISSING
    wandb_filter: Optional[WandbFilterT] = MISSING
    csv_path: Optional[str] = MISSING
    selected_id: Optional[str] = MISSING


@dataclass
class OddImageConfig:
    batch_size: int = MISSING
    select_row_from_index: Optional[int] = MISSING
    select_row_from_current_coefficients: bool = MISSING
    encoder: EncoderOddImageConfig = field(
        default_factory=EncoderOddImageConfig
    )
    checkpoint: Optional[CheckpointOddImageConfig] = None
    optimizer: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class UnpairedCLSDownstreamConfig:
    random_regressor: bool = MISSING
    checkpoint: Any = MISSING
    optimizer: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class DownstreamConfig:
    unpaired_cls: UnpairedCLSDownstreamConfig = field(
        default_factory=UnpairedCLSDownstreamConfig
    )


@dataclass
class ShapesConfig:
    n_train_examples: int = MISSING
    n_val_examples: int = MISSING
    n_test_examples: int = MISSING
    min_scale: int = MISSING
    max_scale: int = MISSING
    min_lightness: int = MISSING
    max_lightness: int = MISSING
    min_hue: int = 0
    max_hue: int = 180
    possible_categories: Optional[List[int]] = None
    min_rotation: float = 0
    max_rotation: float = 360
    min_x: Optional[int] = None
    max_x: Optional[int] = None
    min_y: Optional[int] = None
    max_y: Optional[int] = None
    shapes_color_range: Optional[List[List[int]]] = None


@dataclass
class DatasetsConfig:
    shapes: ShapesConfig = field(default_factory=ShapesConfig)


@dataclass
class LossDefinitionsGWResultsConfig:
    translation: List[str] = MISSING
    contrastive: List[str] = MISSING


@dataclass
class MixLossCoefficientsGWResultConfig:
    translation: float = MISSING
    contrastive: float = MISSING


@dataclass
class GWResultVisualizationConfig:
    saved_figure_path: str = MISSING
    total_num_examples: int = MISSING

    loss_definitions: LossDefinitionsGWResultsConfig = field(
        default_factory=LossDefinitionsGWResultsConfig
    )
    axes: AxesGWResultConfig = field(default_factory=AxesGWResultConfig)
    mix_loss_coefficients: MixLossCoefficientsGWResultConfig = field(
        default_factory=MixLossCoefficientsGWResultConfig
    )
    legend: LegendGWResultConfig = field(default_factory=LegendGWResultConfig)


@dataclass
class BIMConfig:
    debug: bool = MISSING
    devices: int = MISSING
    name: str = MISSING
    accelerator: str = MISSING
    distributed_backend: str = MISSING
    seed: int = MISSING
    fast_dev_run: bool = MISSING
    max_epochs: int = MISSING
    max_steps: int = MISSING

    simple_shapes_path: str = MISSING

    current_dataset: str = MISSING

    img_size: int = MISSING

    n_validation_examples: int = MISSING

    checkpoints_dir: str = MISSING
    resume_from_checkpoint: Any = MISSING
    logger_resume_id: Optional[str] = MISSING

    progress_bar: bool = MISSING

    checkpoint: Any = MISSING

    slurm_id: Optional[str] = MISSING

    run_name: str = MISSING

    loggers: List[LoggerConfig] = MISSING

    domain_loader: Dict[str, Dict[str, Any]] = MISSING

    _code_version: str = "master"
    _script_name: str = "debug"

    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    visualization: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )
    vae: VAEConfig = field(default_factory=VAEConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    global_workspace: GlobalWorkspaceConfig = field(
        default_factory=GlobalWorkspaceConfig
    )
    losses: LossesConfig = field(default_factory=LossesConfig)
    odd_image: OddImageConfig = field(default_factory=OddImageConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
