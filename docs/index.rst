IPython2CWL: Convert Jupyter Notebook to CWL
================================================================================

.. image:: https://badges.gitter.im/ipython2cwl/community.svg
    :target: https://gitter.im/ipython2cwl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. image:: https://travis-ci.com/giannisdoukas/ipython2cwl.svg?branch=master
    :target: https://travis-ci.com/giannisdoukas/ipython2cwl
.. image:: https://coveralls.io/repos/github/giannisdoukas/ipython2cwl/badge.svg?branch=master
    :target: https://coveralls.io/github/giannisdoukas/ipython2cwl?branch=master
.. image:: https://pepy.tech/badge/ipython2cwl/month
    :target: https://github.com/giannisdoukas/ipython2cwl

------------------------------------------------------------------------------------------

IPython2CWL is a tool for converting `IPython <https://ipython.org/>`_ Jupyter Notebooks to
`CWL <https://www.commonwl.org/>`_ Command Line Tools by simply providing typing annotation.

.. code-block:: python

    from ipython2cwl.iotypes import CWLFilePathInput, CWLFilePathOutput
    import csv
    input_filename: 'CWLFilePathInput' = 'data.csv'
    with open(input_filename) as f:
        csv_reader = csv.reader(f)
        data = [line for line in csv_reader]
    number_of_lines = len(data)
    result_file: 'CWLFilePathOutput' = 'number_of_lines.txt'
    with open(result_file, 'w') as f:
        f.write(str(number_of_lines))


------------------------------------------------------------------------------------------

IPython2CWL is based on `repo2docker <https://github.com/jupyter/repo2docker>`_, the same tool
used by `mybinder <https://mybinder.org/>`_. Now, by writing Jupyter Notebook and publishing them, including repo2docker
configuration, the community can not only execute the notebooks remotely but can also use them as steps in scientific
workflows.

* `Install ipython2cwl <https://pypi.org/project/ipython2cwl/>`_: :code:`pip install ipython2cwl`
* Ensure that you have docker running
* Create a directory to store the generated cwl files, for example cwlbuild
* Execute :code:`jupyter repo2cwl https://github.com/giannisdoukas/cwl-annotated-jupyter-notebook.git -o cwlbuild`

HOW IT WORKS?
------------------

IPython2CWL parses each IPython notebook and finds the variables with the typing annotations. For each input variable,
the assigment of that variable will be generalised as a command line argument. Each output variable will be mapped
in the cwl description as an output file.

SUPPORTED TYPES
------------------

.. automodule:: ipython2cwl.iotypes
   :members:


THAT'S COOL! WHAT ABOUT LIST & OPTIONAL ARGUMENTS?
"""""""""""""""""""""""""""""""""""""""""""""""""""

The basic input data types can be combined with the List and Optional annotations. For example, write the following
annotation:

.. code-block:: python
  from typing import List
  file_inputs: List[CWLFilePathInput] = ['data1.txt', 'data2.txt', 'data3.txt']
  example: Optional[CWLStringInput] = None


An ``Optional`` input assigned a literal value (``scale: Optional[CWLFloatInput] = 0.5``) gets that value as
its CWL ``default``; ``None`` keeps the input without default. File inputs never get a default.

An output path written as an f-string of inputs becomes a CWL parameter reference in the ``glob``:
``f"{band}.tif"`` gives ``$(inputs.band).tif`` and ``f"{band_file.stem}_4326.tif"`` gives
``$(inputs.band_file.nameroot)_4326.tif`` (``stem``, ``name`` and ``suffix`` map to ``nameroot``, ``basename`` and
``nameext``). Other placeholders become ``*`` wildcards.


FROM STEP NOTEBOOKS TO A CWL WORKFLOW
"""""""""""""""""""""""""""""""""""""

When a processing chain is split into one notebook per step, :code:`jupyter repo2cwl` can also chain the generated
CommandLineTools into a CWL ``Workflow``. It writes a packed document (``$graph`` holding the workflow and every
tool it runs, ``cwlVersion: v1.2``), usable as an OGC Application Package, next to the per-step tools. By default all the
steps share the image built by repo2docker (see "One image per step" below).

* :code:`jupyter repo2cwl REPO -o cwlbuild --workflow my-workflow`: the steps are the notebooks named
  ``NN_name.ipynb`` (``01_select.ipynb``, ``02_download.ipynb``, ...), in numeric order. An output named ``X``
  feeds the input ``X`` of later steps (the most recent producer wins), an input that no earlier step produces
  becomes a workflow input of the same name, and the workflow outputs are those of the last step.
* :code:`jupyter repo2cwl REPO -o cwlbuild --workflow-manifest workflow.yml`: a YAML manifest lists the steps and
  overrides what the convention cannot express.

.. code-block:: yaml

  id: my-workflow                 # workflow id and file name (my-workflow.cwl)
  label: My workflow
  doc: What it does
  namespaces: {s: https://schema.org/}
  metadata: {s:softwareVersion: 1.0.0}   # document-level fields
  requirements: {}                # added to the workflow requirements
  inputs:                         # declare or override workflow inputs
    bands: {type: string[], default: [red, nir]}
  outputs:                        # default: the outputs of the last step
    result: combine/result
  steps:
    - id: parse_aoi
      run: cwl/parse_aoi.cwl      # a CWL-native tool, embedded in the $graph
    - notebook: steps/01_select.ipynb     # step id: "select"
    - notebook: steps/02_download.ipynb
      in: {band: bands}           # source: workflow input or step/output
      scatter: band
    - notebook: steps/03_combine.ipynb
      image: ghcr.io/org/repo/combine:2.0   # with --per-step-images
      in:
        band_files: download/band_file
        scale: {default: 0.2}     # also: valueFrom, list of sources

Paths in the manifest are relative to the manifest. Obvious type mismatches (array into a single value, File into
a string) stop the generation with a message pointing at the manifest.

To reference a published image instead of the local build, use :code:`--docker-pull ghcr.io/org/repo:tag` (the
image is built under that name; ``DockerRequirement.dockerPull`` is used). Add :code:`--skip-build` to only write
the CWL files.

One image per step
""""""""""""""""""

By default every tool runs in the image built by repo2docker, which holds all the generated scripts. With
:code:`--per-step-images` (it needs :code:`--docker-pull`, which then names the *base* image), the base image holds
the repository environment only, and each notebook tool gets its own thin image on top of it, with only its
script, as in the OSPD Algae Bloom application package. A tool deployed on its own then ships only its code, and
the image digest recorded in provenance identifies the step.

Step images are named from the base image with :code:`--step-image-template` (fields ``{prefix}``,
``{repository}``, ``{name}``, ``{tag}``; default ``{prefix}/{name}:{tag}``): with
:code:`--docker-pull ghcr.io/org/repo/app:1.0`, ``01_select_scene.ipynb`` runs in
``ghcr.io/org/repo/select-scene:1.0``. A manifest step can set its own name with ``image:``. The base image and
the step images are listed in ``images.txt`` in the output directory, in push order.

.. code-block:: bash

  jupyter repo2cwl . -o cwlbuild --workflow-manifest workflow.yml \
    --docker-pull ghcr.io/org/repo/app:1.0 --per-step-images
  xargs -n1 docker push < cwlbuild/images.txt



SEEMS INTERESTING! WHAT ABOUT A DEMO?
----------------------------------------

If you would like to see a demo before you want to start annotating your notebooks check here!
`github.com/giannisdoukas/ipython2cwl-demo <https://github.com/giannisdoukas/ipython2cwl-demo>`_


WHAT IF I WANT TO VALIDATE THAT THE GENERATED SCRIPTS ARE CORRECT?
------------------------------------------------------------------

All the generated scripts are stored in the docker image under the directory :code:`/app/cwl/bin`. You can see the list
of the files by running :code:`docker run [IMAGE_ID] find /app/cwl/bin/ -type f`.
