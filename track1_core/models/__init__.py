"""Trainable LightningModules over the vendored F3Loc observation networks."""

from track1_core.models.visual import (  # noqa: F401
    CompDepthModule,
    MVDepthModule,
    MonoDepthModule,
    build_module,
)
