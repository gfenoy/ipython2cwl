import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import nbformat
import yaml
from git import Repo

from ipython2cwl.cwltoolextractor import AnnotatedIPython2CWLToolConverter
from ipython2cwl.step_images import (
    build_step_image,
    split_image_name,
    step_dockerfile,
    step_image_name,
)
from ipython2cwl.repo2cwl import repo2cwl
from ipython2cwl.workflow import (
    WorkflowBuilder,
    WorkflowError,
    load_manifest,
    numbered_notebooks,
    step_id_from_notebook,
)


def _tool(code: str) -> dict:
    return AnnotatedIPython2CWLToolConverter(code).cwl_command_line_tool()


SELECT = "\n".join(
    [
        "west: CWLFloatInput = 1.0",
        "east: CWLFloatInput = 2.0",
        "stac_item: CWLFilePathOutput = 'item.json'",
    ]
)
DOWNLOAD = "\n".join(
    [
        "stac_item: CWLFilePathInput = 'item.json'",
        "band: CWLStringInput = 'red'",
        "west: CWLFloatInput = 1.0",
        "band_file: CWLFilePathOutput = f'{band}.tif'",
    ]
)
COMBINE = "\n".join(
    [
        "band_files: List[CWLFilePathInput] = ['red.tif']",
        "scale: Optional[CWLFloatInput] = 0.5",
        "result: CWLDirectoryPathOutput = 'outputs'",
    ]
)


class TestOutputGlobsAndDefaults(TestCase):
    def test_fstring_output_glob_uses_parameter_references(self):
        tool = _tool(
            "\n".join(
                [
                    "import os",
                    "from pathlib import Path",
                    "band: CWLStringInput = 'red'",
                    "band_file: CWLFilePathInput = 'red.tif'",
                    "a: CWLFilePathOutput = f'{band}.tif'",
                    "b: CWLFilePathOutput = f'{band_file.stem}_4326.tif'",
                    "c: CWLFilePathOutput = f'{Path(band_file).stem}_x{band_file.suffix}'",
                    "d: CWLFilePathOutput = os.path.join('out', f'{band}_{local}.tif')",
                    "e: CWLFilePathOutput = f'{os.path.basename(band_file)}.aux'",
                ]
            )
        )
        globs = {k: v["outputBinding"]["glob"] for k, v in tool["outputs"].items()}
        self.assertEqual("$(inputs.band).tif", globs["a"])
        self.assertEqual("$(inputs.band_file.nameroot)_4326.tif", globs["b"])
        self.assertEqual(
            "$(inputs.band_file.nameroot)_x$(inputs.band_file.nameext)", globs["c"]
        )
        self.assertEqual("out/$(inputs.band)_*.tif", globs["d"])
        self.assertEqual("$(inputs.band_file.basename).aux", globs["e"])

    def test_optional_literal_becomes_default(self):
        converter = AnnotatedIPython2CWLToolConverter(
            "\n".join(
                [
                    "scale: Optional[CWLFloatInput] = -0.5",
                    "name: Optional[CWLStringInput] = 'x'",
                    "missing: Optional[CWLIntInput] = None",
                    "f: Optional[CWLFilePathInput] = 'data.csv'",
                    "required: CWLFloatInput = 3.0",
                    "print(scale, name, missing, f, required)",
                ]
            )
        )
        inputs = converter.cwl_command_line_tool()["inputs"]
        self.assertEqual(-0.5, inputs["scale"]["default"])
        self.assertEqual("x", inputs["name"]["default"])
        self.assertNotIn("default", inputs["missing"])
        self.assertNotIn("default", inputs["f"])  # File defaults need a location
        self.assertNotIn("default", inputs["required"])
        script = converter._wrap_script_to_method(converter._tree, converter._variables)
        self.assertIn("default=-0.5", script)
        self.assertIn("default='x'", script)


class TestWorkflowBuilder(TestCase):
    maxDiff = None

    def test_step_ids_and_numbered_notebooks(self):
        self.assertEqual("select-scene", step_id_from_notebook("steps/01_select-scene.ipynb"))
        self.assertEqual("plain", step_id_from_notebook("plain.ipynb"))
        self.assertEqual(
            ["b/1_a.ipynb", "a/02_b.ipynb", "a/10_c.ipynb"],
            numbered_notebooks(["/r/a/10_c.ipynb", "/r/x.ipynb", "/r/a/02_b.ipynb", "/r/b/1_a.ipynb"], "/r"),
        )

    def test_convention_wiring(self):
        builder = WorkflowBuilder("wf")
        builder.add_step("01_select.ipynb", _tool(SELECT))
        builder.add_step("02_download.ipynb", _tool(DOWNLOAD))
        document = builder.build()
        self.assertEqual("v1.2", document["cwlVersion"])
        workflow, *tools = document["$graph"]
        self.assertEqual(["select", "download"], [t["id"] for t in tools])
        self.assertNotIn("cwlVersion", tools[0])
        self.assertEqual(
            {"west": {"type": "float"}, "east": {"type": "float"}, "band": {"type": "string"}},
            workflow["inputs"],
        )
        self.assertEqual(
            {"stac_item": "select/stac_item", "band": "band", "west": "west"},
            workflow["steps"]["download"]["in"],
        )
        self.assertEqual(
            {"band_file": {"type": "File", "outputSource": "download/band_file"}},
            workflow["outputs"],
        )
        self.assertNotIn("requirements", workflow)

    def test_manifest_scatter_explicit_in_and_outputs(self):
        manifest = {
            "id": "my-wf",
            "label": "My workflow",
            "namespaces": {"s": "https://schema.org/"},
            "metadata": {"s:softwareVersion": "1.0.0"},
            "inputs": {"bands": {"type": "string[]", "default": ["red", "nir"]}},
            "outputs": {"stac": "combine/result"},
            "steps": [
                {"id": "select", "notebook": "01_select.ipynb"},
                {"id": "download", "notebook": "02_download.ipynb", "in": {"band": "bands"}, "scatter": "band"},
                {"id": "combine", "notebook": "03_combine.ipynb", "in": {"band_files": "download/band_file"}},
            ],
        }
        builder = WorkflowBuilder("ignored", manifest)
        builder.add_step("01_select.ipynb", _tool(SELECT), manifest["steps"][0])
        builder.add_step("02_download.ipynb", _tool(DOWNLOAD), manifest["steps"][1])
        builder.add_step("03_combine.ipynb", _tool(COMBINE), manifest["steps"][2])
        document = builder.build()
        workflow = document["$graph"][0]
        self.assertEqual("my-wf", workflow["id"])
        self.assertEqual("My workflow", workflow["label"])
        self.assertEqual({"s": "https://schema.org/"}, document["$namespaces"])
        self.assertEqual("1.0.0", document["s:softwareVersion"])
        self.assertEqual({"ScatterFeatureRequirement": {}}, workflow["requirements"])
        self.assertEqual("band", workflow["steps"]["download"]["scatter"])
        self.assertEqual({"type": "string[]", "default": ["red", "nir"]}, workflow["inputs"]["bands"])
        self.assertEqual({"type": "float?", "default": 0.5}, workflow["inputs"]["scale"])
        self.assertEqual({"type": "Directory", "outputSource": "combine/result"}, workflow["outputs"]["stac"])

    def test_array_to_scalar_without_scatter_fails(self):
        builder = WorkflowBuilder("wf", {"steps": [{"notebook": "a"}, {"notebook": "b", "scatter": "band"}, {"notebook": "c"}]})
        builder.add_step("01_select.ipynb", _tool(SELECT))
        builder.add_step("02_download.ipynb", _tool(DOWNLOAD), {"scatter": "band"})
        builder.add_step("03_single.ipynb", _tool("band_file: CWLFilePathInput = 'x'\nout: CWLFilePathOutput = 'o'"))
        with self.assertRaises(WorkflowError):
            builder.build()

    def test_file_to_value_fails_and_unknown_references(self):
        builder = WorkflowBuilder("wf")
        builder.add_step("01_select.ipynb", _tool(SELECT))
        builder.add_step("02_bad.ipynb", _tool("stac_item: CWLStringInput = 'x'\no: CWLFilePathOutput = 'o'"))
        with self.assertRaises(WorkflowError):
            builder.build()
        with self.assertRaises(WorkflowError):
            WorkflowBuilder("wf").add_step("01_x.ipynb", _tool(SELECT), {"scatter": "nope"})
        builder = WorkflowBuilder("wf")
        builder.add_step("01_select.ipynb", _tool(SELECT))
        builder.add_step("02_download.ipynb", _tool(DOWNLOAD), {"in": {"stac_item": "select/missing"}})
        with self.assertRaises(WorkflowError):
            builder.build()

    def test_cwl_native_tool_step(self):
        parse_aoi = {
            "cwlVersion": "v1.2",
            "class": "CommandLineTool",
            "id": "parse_aoi",
            "$namespaces": {"eo": "https://example.org/"},
            "baseCommand": "echo",
            "inputs": {"aoi": {"type": "https://example.org/ogc.yaml#BBox"}},
            "outputs": [{"id": "west", "type": "float", "outputBinding": {"outputEval": "$(inputs.aoi.bbox[0])"}}],
        }
        builder = WorkflowBuilder("wf")
        builder.add_step("cwl/parse_aoi.cwl", parse_aoi)
        builder.add_step("01_select.ipynb", _tool(SELECT))
        document = builder.build()
        workflow = document["$graph"][0]
        self.assertEqual({"eo": "https://example.org/"}, document["$namespaces"])
        self.assertEqual("parse_aoi/west", workflow["steps"]["select"]["in"]["west"])
        self.assertEqual({"type": "https://example.org/ogc.yaml#BBox"}, workflow["inputs"]["aoi"])
        self.assertEqual(["parse_aoi", "select"], [t["id"] for t in document["$graph"][1:]])

    def test_manifest_validation(self):
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "workflow.yml")
            for content in ["steps: []", "steps: [{notebook: a, run: b}]", "steps: [{notebook: a, foo: 1}]"]:
                Path(path).write_text(content)
                with self.assertRaises(WorkflowError):
                    load_manifest(path)
        finally:
            shutil.rmtree(directory)


class TestRepo2CWLWorkflow(TestCase):
    """End to end on a small repository, without building the image."""

    maxDiff = None

    def _repo(self) -> str:
        directory = tempfile.mkdtemp()
        steps = os.path.join(directory, "steps")
        os.makedirs(steps)
        for name, code in [("01_select", SELECT), ("02_download", DOWNLOAD), ("03_combine", COMBINE)]:
            notebook = nbformat.v4.new_notebook()
            notebook.cells = [
                nbformat.v4.new_code_cell(
                    "from ipython2cwl.iotypes import CWLFloatInput, CWLFilePathInput, CWLStringInput, "
                    "CWLFilePathOutput, CWLDirectoryPathOutput\nfrom typing import List, Optional"
                ),
                nbformat.v4.new_code_cell(code),
            ]
            nbformat.write(notebook, os.path.join(steps, f"{name}.ipynb"))
        Path(directory, "requirements.txt").write_text("numpy\n")
        Path(directory, "workflow.yml").write_text(
            yaml.safe_dump(
                {
                    "id": "demo",
                    "steps": [
                        {"notebook": "steps/01_select.ipynb"},
                        {"notebook": "steps/02_download.ipynb", "in": {"band": "bands"}, "scatter": "band"},
                        {"notebook": "steps/03_combine.ipynb", "in": {"band_files": "download/band_file"}},
                    ],
                }
            )
        )
        repo = Repo.init(directory)
        repo.git.add(A=True)
        repo.index.commit("initial commit")
        return directory

    def test_tools_and_workflow_from_repo(self):
        directory = self._repo()
        output = tempfile.mkdtemp()
        try:
            self.assertEqual(
                0,
                repo2cwl(
                    [directory, "-o", output, "--workflow-manifest", "workflow.yml",
                     "--docker-pull", "ghcr.io/example/demo:1.0", "--skip-build"]
                ),
            )
            self.assertEqual(
                {"steps_01_select.cwl", "steps_02_download.cwl", "steps_03_combine.cwl", "demo.cwl",
                 "images.txt"},
                set(os.listdir(output)),
            )
            with open(os.path.join(output, "demo.cwl")) as f:
                document = yaml.safe_load(f)
            workflow, *tools = document["$graph"]
            self.assertEqual(["select", "download", "combine"], list(workflow["steps"]))
            self.assertEqual("/app/cwl/bin/steps/02_download", tools[1]["baseCommand"])
            self.assertEqual({"DockerRequirement": {"dockerPull": "ghcr.io/example/demo:1.0"}}, tools[1]["hints"])
            self.assertEqual("$(inputs.band).tif", tools[1]["outputs"]["band_file"]["outputBinding"]["glob"])
            self.assertEqual({"type": "string[]"}, workflow["inputs"]["bands"])
            self.assertEqual({"result": {"type": "Directory", "outputSource": "combine/result"}}, workflow["outputs"])

            # convention mode on the same repository: names only, no scatter; band_files
            # is not produced by any step, so it becomes a workflow input
            self.assertEqual(
                0,
                repo2cwl([directory, "-o", output, "--workflow", "conv",
                          "--docker-pull", "x:1", "--skip-build"]),
            )
            with open(os.path.join(output, "conv.cwl")) as f:
                workflow = yaml.safe_load(f)["$graph"][0]
            self.assertEqual(["select", "download", "combine"], list(workflow["steps"]))
            self.assertEqual("select/stac_item", workflow["steps"]["download"]["in"]["stac_item"])
            self.assertEqual({"type": "File[]"}, workflow["inputs"]["band_files"])
        finally:
            shutil.rmtree(directory)
            shutil.rmtree(output)

    def test_manifest_lookup(self):
        directory = self._repo()
        output = tempfile.mkdtemp()
        elsewhere = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            # missing manifest: fails before building the image (no Docker needed here)
            with self.assertRaises(WorkflowError) as error:
                repo2cwl([directory, "-o", output, "--workflow-manifest", "nope.yml"])
            self.assertIn("not found", str(error.exception))

            # manifest outside the repository, given as a local path: its paths are
            # relative to the repository root
            shutil.move(os.path.join(directory, "workflow.yml"), os.path.join(elsewhere, "wf.yml"))
            os.chdir(elsewhere)
            self.assertEqual(
                0,
                repo2cwl([directory, "-o", output, "--workflow-manifest", "wf.yml",
                          "--docker-pull", "x:1", "--skip-build"]),
            )
            self.assertIn("demo.cwl", os.listdir(output))

            # a step file missing from the repository is reported before the build
            with open("wf.yml") as f:
                manifest = yaml.safe_load(f)
            manifest["steps"][0]["notebook"] = "steps/99_missing.ipynb"
            with open("wf.yml", "w") as f:
                yaml.safe_dump(manifest, f)
            with self.assertRaises(WorkflowError) as error:
                repo2cwl([directory, "-o", output, "--workflow-manifest", "wf.yml"])
            self.assertIn("steps/99_missing.ipynb", str(error.exception))
        finally:
            os.chdir(cwd)
            for d in (directory, output, elsewhere):
                shutil.rmtree(d)


class TestStepImages(TestCase):
    maxDiff = None

    def test_names(self):
        self.assertEqual(("ghcr.io/o/r/app", "1.0"), split_image_name("ghcr.io/o/r/app:1.0"))
        self.assertEqual(("localhost:5000/app", "latest"), split_image_name("localhost:5000/app"))
        self.assertEqual(
            "ghcr.io/o/r/select-scene:1.0", step_image_name("ghcr.io/o/r/app:1.0", "select_scene")
        )
        self.assertEqual("select-scene:dev", step_image_name("app:dev", "select_scene"))
        self.assertEqual(
            "ghcr.io/o/r/app-select-scene:1.0",
            step_image_name("ghcr.io/o/r/app:1.0", "select_scene", "{repository}-{name}:{tag}"),
        )

    def test_dockerfile(self):
        dockerfile = step_dockerfile(
            "base:1", "01_select", "/app/cwl/bin/steps/01_select", "jovyan", {"a.b": 'x"y'}
        )
        self.assertEqual(
            "FROM base:1\nUSER root\n"
            "COPY 01_select /app/cwl/bin/steps/01_select\n"
            "RUN chmod 0755 /app/cwl/bin/steps/01_select\nUSER jovyan\n"
            'LABEL "a.b"="x\\"y"\n',
            dockerfile,
        )

    def test_build_with_docker_client(self):
        calls = {}

        class Images:
            def get(self, name):
                calls["get"] = name
                return type("Image", (), {"attrs": {"Config": {"User": "1000"}}})()

            def build(self, path, tag, rm):
                calls["tag"] = tag
                calls["files"] = sorted(os.listdir(path))
                calls["dockerfile"] = Path(path, "Dockerfile").read_text()

        client = type("Client", (), {"images": Images()})()
        directory = tempfile.mkdtemp()
        try:
            script = Path(directory, "01_select")
            script.write_text("print(1)")
            build_step_image("base:1", "step:1", str(script), "/app/cwl/bin/01_select", {}, client)
        finally:
            shutil.rmtree(directory)
        self.assertEqual("base:1", calls["get"])
        self.assertEqual("step:1", calls["tag"])
        self.assertEqual(["01_select", "Dockerfile"], calls["files"])
        self.assertIn("USER 1000", calls["dockerfile"])

    def test_repo2cwl_per_step_images(self):
        directory = TestRepo2CWLWorkflow()._repo()
        with open(os.path.join(directory, "workflow.yml")) as f:
            manifest = yaml.safe_load(f)
        manifest["steps"][2]["image"] = "ghcr.io/example/combine-custom:2.0"
        with open(os.path.join(directory, "workflow.yml"), "w") as f:
            yaml.safe_dump(manifest, f)
        output = tempfile.mkdtemp()
        try:
            self.assertEqual(
                0,
                repo2cwl([directory, "-o", output, "--workflow-manifest", "workflow.yml",
                          "--docker-pull", "ghcr.io/example/demo:1.0", "--skip-build",
                          "--per-step-images"]),
            )
            with open(os.path.join(output, "demo.cwl")) as f:
                tools = yaml.safe_load(f)["$graph"][1:]
            self.assertEqual(
                ["ghcr.io/example/select:1.0", "ghcr.io/example/download:1.0",
                 "ghcr.io/example/combine-custom:2.0"],
                [t["hints"]["DockerRequirement"]["dockerPull"] for t in tools],
            )
            self.assertEqual(
                ["ghcr.io/example/demo:1.0", "ghcr.io/example/combine-custom:2.0",
                 "ghcr.io/example/download:1.0", "ghcr.io/example/select:1.0"],
                Path(output, "images.txt").read_text().split(),
            )
            with self.assertRaises(SystemExit):
                repo2cwl([directory, "-o", output, "--per-step-images"])
        finally:
            shutil.rmtree(directory)
            shutil.rmtree(output)
