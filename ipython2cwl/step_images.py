"""One container image per notebook tool.

repo2docker builds a single *base* image from the repository environment
(requirements.txt, environment.yml, ...). With per-step images, each notebook tool gets
its own thin image on top of it, holding only its generated script, in the way of the
OSPD Algae Bloom application package (a shared base image, one image per process). Each
CommandLineTool then references its own image, so that a tool deployed on its own ships
only its code, and the image digest recorded in provenance identifies the step.
"""
import logging
import os
import re
import shutil
import tempfile
from typing import Dict, Optional, Tuple

logger = logging.getLogger("repo2cwl")

DEFAULT_TEMPLATE = "{prefix}/{name}:{tag}"


def split_image_name(image: str) -> Tuple[str, str]:
    """``ghcr.io/org/repo/app:1.0`` -> (``ghcr.io/org/repo/app``, ``1.0``).

    A ``:`` in the registry part (``localhost:5000/app``) is not taken for a tag."""
    last = image.rsplit("/", 1)[-1]
    if ":" in last:
        repository, tag = image.rsplit(":", 1)
        return repository, tag
    return image, "latest"


def step_image_name(base_image: str, name: str, template: Optional[str] = None) -> str:
    """Name of the image of a step, from the base image name.

    Template fields: ``{repository}`` (base image without tag), ``{prefix}`` (base
    repository without its last component), ``{name}`` (step name), ``{tag}`` (base tag).
    The default, ``{prefix}/{name}:{tag}``, turns ``ghcr.io/org/repo/app:1.0`` and
    ``select_scene`` into ``ghcr.io/org/repo/select-scene:1.0``."""
    repository, tag = split_image_name(base_image)
    prefix = repository.rsplit("/", 1)[0] if "/" in repository else ""
    name = re.sub(r"[^a-z0-9._-]+", "-", name.lower().replace("_", "-")).strip("-.")
    template = template or (DEFAULT_TEMPLATE if prefix else "{name}:{tag}")
    return template.format(repository=repository, prefix=prefix, name=name, tag=tag)


def step_dockerfile(
    base_image: str,
    script_file: str,
    target: str,
    user: Optional[str],
    labels: Dict[str, str],
) -> str:
    """Dockerfile of a step image: the base image plus one executable script.

    The classic builder used through docker-py has no ``COPY --chmod``: the copy and the
    chmod run as root, then the base image user is restored."""
    lines = [
        f"FROM {base_image}",
        "USER root",
        f"COPY {script_file} {target}",
        f"RUN chmod 0755 {target}",
    ]
    if user:
        lines.append(f"USER {user}")
    for key, value in sorted(labels.items()):
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'LABEL "{key}"="{escaped}"')
    return "\n".join(lines) + "\n"


def build_step_image(
    base_image: str,
    image: str,
    script_path: str,
    target: str,
    labels: Dict[str, str],
    client=None,
) -> str:
    """Build the image of one step with docker-py; returns its name."""
    if client is None:
        import docker  # type: ignore  # installed with jupyter-repo2docker

        client = docker.from_env()
    user = client.images.get(base_image).attrs.get("Config", {}).get("User") or None
    context = tempfile.mkdtemp(prefix="repo2cwl_step_")
    try:
        script_file = os.path.basename(script_path)
        shutil.copy(script_path, os.path.join(context, script_file))
        with open(os.path.join(context, "Dockerfile"), "w") as f:
            f.write(step_dockerfile(base_image, script_file, target, user, labels))
        logger.info(f"Building step image {image} (from {base_image})")
        client.images.build(path=context, tag=image, rm=True)
    finally:
        shutil.rmtree(context)
    return image
