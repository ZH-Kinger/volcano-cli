"""volcano — a tiny CLI/SDK to submit training jobs as Volcano Jobs.

Usage as an SDK::

    import volcano

    job = volcano.submit(
        name="my-run",
        team="wuji-rl",
        image="wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/<你的镜像>:<tag>",
        gpus=8,
        command="python train.py",
        data="datasets/imagenet",
    )
    print(volcano.list_jobs(team="wuji-rl"))

The same functions back the ``volcano`` command line tool (see ``volcano.cli``).
"""

from __future__ import annotations

from .job import build_volcano_job
from .sdk import kill, list_images, list_jobs, logs, save_image, status, submit

__all__ = [
    "submit",
    "list_jobs",
    "list_images",
    "kill",
    "status",
    "logs",
    "save_image",
    "build_volcano_job",
]

__version__ = "0.12.0"
