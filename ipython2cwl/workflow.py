"""Assemble per-notebook CWL CommandLineTools into a packed CWL Workflow.

Each step notebook is converted, as before, into its own CommandLineTool. This module
wires those tools together into a ``Workflow`` and packs the workflow and every tool it
runs into a single ``$graph`` document (the shape expected by OGC Application Packages,
OGC Best Practice 20-089).

Wiring follows a naming convention, which an optional manifest can override:

* steps run in the order given by the manifest or, without a manifest, in the order of
  the numbered notebook file names (``01_select.ipynb``, ``02_download.ipynb``, ...);
* an output named ``X`` of a step feeds the input named ``X`` of any later step (the
  most recent producer wins);
* a step input that no earlier step produces becomes a workflow input of the same name
  (steps sharing an unwired input name share the workflow input);
* the workflow outputs are the outputs of the last step.

The manifest (YAML) can set the workflow ``id``/``label``/``doc``, add requirements,
namespaces and document-level metadata, declare or override workflow ``inputs`` and
``outputs``, and for each step give an explicit ``in`` mapping, a ``scatter``, or a
CWL-native tool (``run:``) instead of a notebook::

    id: my-workflow
    steps:
      - id: parse_aoi
        run: cwl/parse_aoi.cwl
      - id: download
        notebook: steps/02_download.ipynb
        in:
          band: bands            # workflow input "bands"
        scatter: band
      - id: compute
        notebook: steps/03_compute.ipynb
        in:
          bands: download/band_file
    outputs:
      result: compute/result
"""
import copy
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger("repo2cwl")

CWL_VERSION = "v1.2"
NUMBERED_NOTEBOOK = re.compile(r"^(\d+)[_\-](.+)\.ipynb$")
MANIFEST_STEP_KEYS = {"id", "notebook", "run", "in", "scatter", "scatterMethod", "label", "doc", "image"}


class WorkflowError(ValueError):
    """Raised when the steps cannot be wired into a valid workflow."""


# --------------------------------------------------------------------------- helpers


def _as_mapping(section: Union[None, Dict, List]) -> Dict[str, Dict]:
    """Normalise a CWL inputs/outputs section (map or list form) to a mapping."""
    if section is None:
        return {}
    if isinstance(section, dict):
        result = {}
        for name, value in section.items():
            result[name] = value if isinstance(value, dict) else {"type": value}
        return result
    result = {}
    for item in section:
        item = dict(item)
        name = item.pop("id").lstrip("#").split("/")[-1]
        result[name] = item
    return result


def _is_optional(cwl_type: Any) -> bool:
    if isinstance(cwl_type, str):
        return cwl_type.endswith("?")
    if isinstance(cwl_type, list):
        return "null" in cwl_type
    return False


def _strip_optional(cwl_type: Any) -> Any:
    if isinstance(cwl_type, str) and cwl_type.endswith("?"):
        return cwl_type[:-1]
    if isinstance(cwl_type, list):
        non_null = [t for t in cwl_type if t != "null"]
        return non_null[0] if len(non_null) == 1 else non_null
    return cwl_type


def _array_of(cwl_type: Any) -> Any:
    cwl_type = _strip_optional(cwl_type)
    if isinstance(cwl_type, str):
        return cwl_type + "[]"
    return {"type": "array", "items": cwl_type}


def _item_of(cwl_type: Any) -> Any:
    """Item type of an array type, or None when the type is not an array."""
    cwl_type = _strip_optional(cwl_type)
    if isinstance(cwl_type, str) and cwl_type.endswith("[]"):
        return cwl_type[:-2]
    if isinstance(cwl_type, dict) and cwl_type.get("type") == "array":
        return cwl_type.get("items")
    return None


def _describe(cwl_type: Any) -> str:
    return cwl_type if isinstance(cwl_type, str) else yaml.safe_dump(cwl_type, default_flow_style=True).strip()


def _check_compatible(source_type: Any, sink_type: Any, where: str) -> None:
    """Fail on obvious mismatches (array vs scalar, File vs Directory vs value)."""
    if source_type is None or sink_type is None:
        return
    source_item, sink_item = _item_of(source_type), _item_of(sink_type)
    if (source_item is None) != (sink_item is None):
        raise WorkflowError(
            f"{where}: cannot connect {_describe(source_type)} to {_describe(sink_type)} "
            "(array vs single value); add an explicit `in` mapping or a `scatter` "
            "in the workflow manifest"
        )
    if source_item is not None:
        source_type, sink_type = source_item, sink_item
    source_type, sink_type = _strip_optional(source_type), _strip_optional(sink_type)
    if not (isinstance(source_type, str) and isinstance(sink_type, str)):
        return
    file_like = {"File", "Directory"}
    if source_type != sink_type and (source_type in file_like or sink_type in file_like):
        raise WorkflowError(
            f"{where}: cannot connect {source_type} to {sink_type}; add an explicit "
            "`in` mapping in the workflow manifest"
        )
    if source_type != sink_type:
        logger.warning(f"{where}: connecting {source_type} to {sink_type}")


def step_id_from_notebook(notebook_path: str) -> str:
    """``steps/01_select-scene.ipynb`` -> ``select-scene``."""
    name = os.path.basename(notebook_path)
    match = NUMBERED_NOTEBOOK.match(name)
    stem = match.group(2) if match else name[: -len(".ipynb")]
    return re.sub(r"[^A-Za-z0-9_\-]", "_", stem)


def numbered_notebooks(notebook_paths: List[str], root: str) -> List[str]:
    """Numbered notebooks (``NN_name.ipynb``) sorted by number then path, relative to root."""
    selected = []
    for path in notebook_paths:
        match = NUMBERED_NOTEBOOK.match(os.path.basename(path))
        if match:
            selected.append((int(match.group(1)), os.path.relpath(path, root)))
    return [path for _, path in sorted(selected)]


def load_manifest(path: str) -> Dict:
    with open(path) as f:
        manifest = yaml.safe_load(f) or {}
    if not isinstance(manifest, dict):
        raise WorkflowError(f"{path}: the workflow manifest must be a YAML mapping")
    steps = manifest.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        raise WorkflowError(f"{path}: `steps` must be a non-empty list")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise WorkflowError(f"{path}: step #{index + 1} must be a mapping")
        if ("notebook" in step) == ("run" in step):
            raise WorkflowError(
                f"{path}: step #{index + 1} needs exactly one of `notebook` or `run`"
            )
        unknown = set(step) - MANIFEST_STEP_KEYS
        if unknown:
            raise WorkflowError(f"{path}: step #{index + 1}: unknown keys {sorted(unknown)}")
    return manifest


# --------------------------------------------------------------------------- builder


class _Step:
    def __init__(self, step_id: str, tool_id: str, tool: Dict, spec: Dict):
        self.id = step_id
        self.tool_id = tool_id
        self.tool = tool
        self.inputs = _as_mapping(tool.get("inputs"))
        self.outputs = _as_mapping(tool.get("outputs"))
        scatter = spec.get("scatter")
        self.scatter: List[str] = [scatter] if isinstance(scatter, str) else list(scatter or [])
        self.scatter_method: Optional[str] = spec.get("scatterMethod")
        self.in_overrides: Dict = spec.get("in") or {}
        self.label = spec.get("label")
        self.doc = spec.get("doc")
        for name in self.scatter:
            if name not in self.inputs:
                raise WorkflowError(f"step {step_id}: cannot scatter over unknown input `{name}`")
        for name in self.in_overrides:
            if name not in self.inputs:
                raise WorkflowError(f"step {step_id}: `in` refers to unknown input `{name}`")
        if len(self.scatter) > 1 and not self.scatter_method:
            self.scatter_method = "dotproduct"

    def output_type(self, name: str) -> Any:
        cwl_type = self.outputs[name].get("type")
        return _array_of(cwl_type) if self.scatter else cwl_type

    def sink_type(self, name: str) -> Any:
        """Type expected from the source of input ``name`` (arrays when scattered)."""
        cwl_type = self.inputs[name].get("type")
        return _array_of(cwl_type) if name in self.scatter else cwl_type


class WorkflowBuilder:
    """Collects tools and steps, then produces the packed workflow document."""

    def __init__(self, workflow_id: str, manifest: Optional[Dict] = None):
        self.manifest = manifest or {}
        self.workflow_id = self.manifest.get("id", workflow_id)
        self.steps: List[_Step] = []
        self.tools: Dict[str, Dict] = {}  # tool id -> tool
        self._tool_sources: Dict[str, str] = {}  # source path -> tool id

    # -- tools ---------------------------------------------------------------------

    def add_tool(self, source: str, tool: Dict, preferred_id: str) -> str:
        """Register a tool (once per source) and return its id in the graph."""
        if source in self._tool_sources:
            return self._tool_sources[source]
        tool_id = preferred_id
        suffix = 2
        while tool_id in self.tools or tool_id == self.workflow_id:
            tool_id = f"{preferred_id}_{suffix}"
            suffix += 1
        self.tools[tool_id] = tool
        self._tool_sources[source] = tool_id
        return tool_id

    def add_step(self, source: str, tool: Dict, spec: Optional[Dict] = None) -> None:
        spec = spec or {}
        if source.endswith(".ipynb"):
            source_id = step_id_from_notebook(source)
        else:
            source_id = re.sub(r"[^A-Za-z0-9_\-]", "_", os.path.splitext(os.path.basename(source))[0])
        tool_id = self.add_tool(source, tool, str(tool.get("id", "")).lstrip("#") or source_id)
        step_id = spec.get("id") or source_id
        if any(s.id == step_id for s in self.steps):
            raise WorkflowError(f"duplicate step id `{step_id}`; set distinct `id`s in the manifest")
        self.steps.append(_Step(step_id, tool_id, tool, spec))

    # -- wiring --------------------------------------------------------------------

    def _producer(self, name: str, before: int) -> Optional[Tuple[_Step, str]]:
        for step in reversed(self.steps[:before]):
            if name in step.outputs:
                return step, name
        return None

    def _resolve_source(self, source: str, before: int, where: str) -> Tuple[str, Any, bool]:
        """Return (source, type, is_workflow_input) for an explicit source string."""
        if "/" in source:
            step_id, output = source.split("/", 1)
            for step in self.steps[:before]:
                if step.id == step_id:
                    if output not in step.outputs:
                        raise WorkflowError(f"{where}: step `{step_id}` has no output `{output}`")
                    return source, step.output_type(output), False
            raise WorkflowError(f"{where}: `{source}` does not refer to an earlier step")
        return source, None, True

    def build(self) -> Dict:
        if not self.steps:
            raise WorkflowError("no step to assemble into a workflow")

        declared_inputs = _as_mapping(self.manifest.get("inputs"))
        workflow_inputs: Dict[str, Dict] = {}
        workflow_steps: Dict[str, Dict] = {}
        uses_scatter = uses_value_from = uses_multiple_sources = False

        def add_workflow_input(name: str, definition: Dict, where: str) -> None:
            if name in declared_inputs:
                if name not in workflow_inputs:
                    workflow_inputs[name] = copy.deepcopy(declared_inputs[name])
                    workflow_inputs[name].setdefault("type", definition.get("type"))
                return
            if name in workflow_inputs:
                existing = workflow_inputs[name].get("type")
                if _strip_optional(existing) != _strip_optional(definition.get("type")):
                    raise WorkflowError(
                        f"{where}: workflow input `{name}` is used with types "
                        f"{_describe(existing)} and {_describe(definition.get('type'))}; "
                        "rename one of them with an explicit `in` mapping"
                    )
                if not _is_optional(definition.get("type")):
                    workflow_inputs[name]["type"] = definition.get("type")
                return
            workflow_inputs[name] = definition

        for index, step in enumerate(self.steps):
            step_in: Dict[str, Any] = {}
            for name, tool_input in step.inputs.items():
                where = f"step {step.id}, input {name}"
                sink_type = step.sink_type(name)
                override = step.in_overrides.get(name)
                if isinstance(override, str):
                    override = {"source": override}

                if override is not None:
                    entry = {k: v for k, v in override.items() if k in ("source", "default", "valueFrom", "linkMerge")}
                    uses_value_from |= "valueFrom" in entry
                    sources = entry.get("source")
                    if isinstance(sources, list):
                        uses_multiple_sources = True
                        for source in sources:
                            self._wire_explicit(source, index, where, None, add_workflow_input, tool_input)
                    elif sources is not None:
                        self._wire_explicit(sources, index, where, sink_type, add_workflow_input, tool_input,
                                            check=("valueFrom" not in entry))
                    step_in[name] = entry["source"] if list(entry) == ["source"] else entry
                    continue

                producer = self._producer(name, index)
                if producer is not None:
                    producer_step, output = producer
                    _check_compatible(producer_step.output_type(output), sink_type, where)
                    step_in[name] = f"{producer_step.id}/{output}"
                    continue

                definition = {"type": sink_type}
                for key in ("label", "doc"):
                    if key in tool_input:
                        definition[key] = tool_input[key]
                if "default" in tool_input and name not in step.scatter:
                    definition["default"] = tool_input["default"]
                add_workflow_input(name, definition, where)
                step_in[name] = name

            workflow_step: Dict[str, Any] = {"run": f"#{step.tool_id}", "in": step_in, "out": list(step.outputs)}
            if step.label:
                workflow_step["label"] = step.label
            if step.doc:
                workflow_step["doc"] = step.doc
            if step.scatter:
                uses_scatter = True
                workflow_step["scatter"] = step.scatter[0] if len(step.scatter) == 1 else step.scatter
                if step.scatter_method:
                    workflow_step["scatterMethod"] = step.scatter_method
            workflow_steps[step.id] = workflow_step

        # declared inputs that no step refers to explicitly or by convention are kept
        for name, definition in declared_inputs.items():
            if name not in workflow_inputs:
                workflow_inputs[name] = copy.deepcopy(definition)

        workflow_outputs = self._workflow_outputs()

        requirements: Dict[str, Any] = {}
        if uses_scatter:
            requirements["ScatterFeatureRequirement"] = {}
        if uses_value_from:
            requirements["StepInputExpressionRequirement"] = {}
            requirements["InlineJavascriptRequirement"] = {}
        if uses_multiple_sources:
            requirements["MultipleInputFeatureRequirement"] = {}
        requirements.update(copy.deepcopy(self.manifest.get("requirements") or {}))

        workflow: Dict[str, Any] = {"class": "Workflow", "id": self.workflow_id}
        for key in ("label", "doc"):
            if key in self.manifest:
                workflow[key] = self.manifest[key]
        if requirements:
            workflow["requirements"] = requirements
        workflow["inputs"] = workflow_inputs
        workflow["outputs"] = workflow_outputs
        workflow["steps"] = workflow_steps

        return self._pack(workflow)

    def _wire_explicit(self, source, index, where, sink_type, add_workflow_input, tool_input, check=True):
        source, source_type, is_workflow_input = self._resolve_source(source, index, where)
        if is_workflow_input:
            definition = {"type": sink_type if sink_type is not None else tool_input.get("type")}
            for key in ("label", "doc"):
                if key in tool_input:
                    definition[key] = tool_input[key]
            add_workflow_input(source, definition, where)
        elif check:
            _check_compatible(source_type, sink_type, where)

    def _workflow_outputs(self) -> Dict[str, Dict]:
        declared = self.manifest.get("outputs")
        outputs: Dict[str, Dict] = {}
        if declared:
            for name, value in declared.items():
                definition = {"outputSource": value} if isinstance(value, str) else dict(value)
                if "source" in definition:
                    definition["outputSource"] = definition.pop("source")
                source = definition.get("outputSource")
                if not isinstance(source, str) or "/" not in source:
                    raise WorkflowError(f"workflow output `{name}` needs a `step/output` source")
                _, source_type, _ = self._resolve_source(source, len(self.steps), f"workflow output {name}")
                definition.setdefault("type", source_type)
                outputs[name] = {"type": definition.pop("type"), **definition}
            return outputs
        last = self.steps[-1]
        for name in last.outputs:
            outputs[name] = {"type": last.output_type(name), "outputSource": f"{last.id}/{name}"}
        return outputs

    def _pack(self, workflow: Dict) -> Dict:
        namespaces: Dict[str, str] = {}
        graph: List[Dict] = [workflow]
        for tool_id, tool in self.tools.items():
            tool = copy.deepcopy(tool)
            tool.pop("cwlVersion", None)
            namespaces.update(tool.pop("$namespaces", {}) or {})
            tool.pop("$schemas", None)
            graph.append({"class": tool.pop("class"), "id": tool_id, **{k: v for k, v in tool.items() if k != "id"}})
        namespaces.update(self.manifest.get("namespaces") or {})

        document: Dict[str, Any] = {"cwlVersion": CWL_VERSION}
        if namespaces:
            document["$namespaces"] = namespaces
        if self.manifest.get("schemas"):
            document["$schemas"] = self.manifest["schemas"]
        document.update(copy.deepcopy(self.manifest.get("metadata") or {}))
        document["$graph"] = graph
        return document


def load_cwl_tool(path: str) -> Dict:
    """Load a CWL-native tool referenced by ``run:`` in the manifest."""
    with open(path) as f:
        tool = yaml.safe_load(f)
    if not isinstance(tool, dict) or tool.get("class") not in ("CommandLineTool", "ExpressionTool", "Workflow"):
        raise WorkflowError(f"{path}: expected a single CWL process (no $graph)")
    return tool
