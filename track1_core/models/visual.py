"""Training wrappers for the F3Loc visual observation networks.

Upstream released Lightning wrappers but not the training script, and the
wrappers log without ``sync_dist`` and expose no hyperparameters or schedule.
These modules re-declare the same losses (L1, plus the paper's cosine shape
term for the monocular net) with DDP-correct logging and checkpointable
hyperparameters.

Submodule attribute names deliberately match upstream (``encoder``, ``net``,
``comp_d_net``) so checkpoints written here load straight into
``depth_net_pl`` / ``mv_depth_net_pl`` / ``comp_d_net_pl`` and can be evaluated
with the vendored evaluation code.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from track1_core import _vendor  # noqa: F401  (puts vendored f3loc on sys.path)
from modules.comp.comp_d_net import comp_d_net
from modules.mono.depth_net import depth_net
from modules.mv.mv_depth_net import mv_depth_net


def depth_losses(pred: torch.Tensor, target: torch.Tensor, shape_weight: float | None):
    """L1 plus the optional cosine shape term, as described in the paper."""
    l1 = F.l1_loss(pred, target)
    if shape_weight is None:
        return l1, l1, None
    shape = shape_weight * (1 - F.cosine_similarity(pred, target, dim=-1).mean())
    return l1 + shape, l1, shape


class _VisualDepthModule(pl.LightningModule):
    """Shared optimisation, logging, and scheduling for the three networks."""

    def __init__(
        self,
        lr: float,
        shape_loss_weight: float | None,
        warmup_steps: int,
        max_steps: int,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.lr = lr
        self.shape_loss_weight = shape_loss_weight
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        # "none" reproduces upstream's configure_optimizers, which returns a
        # bare Adam with no warmup and no decay.
        self.schedule = schedule

    def forward_depth(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prediction, target) for the concrete network."""
        raise NotImplementedError

    def _step(self, batch, stage: str) -> torch.Tensor:
        pred, target = self.forward_depth(batch)
        loss, l1, shape = depth_losses(pred, target, self.shape_loss_weight)
        on_step = stage == "train"
        sync = stage != "train"
        self.log(f"l1_loss-{stage}", l1, on_step=on_step, on_epoch=True, sync_dist=sync)
        if shape is not None:
            self.log(f"shape_loss-{stage}", shape, on_step=on_step, on_epoch=True, sync_dist=sync)
        self.log(
            f"loss-{stage}",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr)
        if self.schedule == "none" or self.max_steps <= 0:
            return optimizer

        warmup = max(self.warmup_steps, 1)
        total = max(self.max_steps, warmup + 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / (total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class MonoDepthModule(_VisualDepthModule):
    """Monocular structural depth network (upstream ``depth_net_pl``)."""

    def __init__(
        self,
        lr: float = 1e-4,
        shape_loss_weight: float | None = 1.0,
        d_min: float = 0.1,
        d_max: float = 15.0,
        d_hyp: float = -0.2,
        D: int = 128,
        F_W: float = 3 / 8,
        warmup_steps: int = 500,
        max_steps: int = 0,
        schedule: str = "cosine",
    ) -> None:
        super().__init__(lr, shape_loss_weight, warmup_steps, max_steps, schedule)
        self.save_hyperparameters()
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.F_W = F_W
        self.encoder = depth_net(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)

    def forward_depth(self, batch):
        rays, _attn, _prob = self.encoder(batch["img"], batch.get("mask"))
        return rays, batch["gt_rays"]


class MVDepthModule(_VisualDepthModule):
    """Multi-view structural depth network (upstream ``mv_depth_net_pl``)."""

    def __init__(
        self,
        lr: float = 1e-4,
        shape_loss_weight: float | None = None,
        d_min: float = 0.1,
        d_max: float = 15.0,
        d_hyp: float = -0.2,
        D: int = 128,
        F_W: float = 3 / 8,
        warmup_steps: int = 500,
        max_steps: int = 0,
        schedule: str = "cosine",
    ) -> None:
        super().__init__(lr, shape_loss_weight, warmup_steps, max_steps, schedule)
        self.save_hyperparameters()
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.F_W = F_W
        self.net = mv_depth_net(D=D, d_min=d_min, d_max=d_max, d_hyp=d_hyp, F_W=F_W)

    def forward_depth(self, batch):
        return self.net(batch)["d"], batch["ref_depth"]


class CompDepthModule(_VisualDepthModule):
    """Complementary selector over frozen mono/mv nets (upstream ``comp_d_net_pl``).

    ``comp_d_net`` freezes both observation networks in its constructor, so only
    the selector MLP is optimised here. The frozen nets are additionally pinned
    to eval mode: their BatchNorm running statistics must not drift, which is the
    same reason upstream evaluation calls ``comp_net.eval()``.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        shape_loss_weight: float | None = None,
        L: int = 3,
        d_min: float = 0.1,
        d_max: float = 15.0,
        d_hyp: float = -0.2,
        D: int = 128,
        F_W: float = 3 / 8,
        use_pred: bool = True,
        warmup_steps: int = 200,
        max_steps: int = 0,
        schedule: str = "cosine",
        mono_ckpt: str | None = None,
        mv_ckpt: str | None = None,
    ) -> None:
        super().__init__(lr, shape_loss_weight, warmup_steps, max_steps, schedule)
        self.save_hyperparameters()
        self.L = L
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.F_W = F_W

        mono_net = depth_net(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
        mv_net = mv_depth_net(D=D, d_min=d_min, d_max=d_max, d_hyp=d_hyp, F_W=F_W)
        if mono_ckpt:
            _load_submodule(mono_net, mono_ckpt, prefix="encoder.")
        if mv_ckpt:
            _load_submodule(mv_net, mv_ckpt, prefix="net.")

        self.comp_d_net = comp_d_net(
            mv_net=mv_net,
            mono_net=mono_net,
            L=L,
            C=64,
            D=D,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            use_pred=use_pred,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # keep the frozen observation networks (and their BatchNorm stats) fixed
        self.comp_d_net.mv_net.eval()
        self.comp_d_net.mono_net.eval()
        return self

    def forward_depth(self, batch):
        return self.comp_d_net(batch)["d_comp"], batch["ref_depth"]


def _load_submodule(module: torch.nn.Module, ckpt_path: str, prefix: str) -> None:
    """Load one sub-network out of a Lightning checkpoint saved by this package."""
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("state_dict", state)
    stripped = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not stripped:
        raise KeyError(f"no parameters with prefix {prefix!r} in {ckpt_path}")
    module.load_state_dict(stripped, strict=True)


def build_module(net_type: str, **kwargs):
    if net_type == "mono":
        return MonoDepthModule(**kwargs)
    if net_type == "mv":
        return MVDepthModule(**kwargs)
    if net_type == "comp":
        return CompDepthModule(**kwargs)
    raise ValueError(f"unknown net_type {net_type!r}")
