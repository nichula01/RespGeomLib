<div align="center">

# RespGeomLib

### A reproducible parametric engine for generating analysis-ready human airway lumen geometry

<p>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/status-research%20prototype-orange" alt="Status">
  <img src="https://img.shields.io/badge/paper-arXiv%20soon-red" alt="Paper coming soon">
  <img src="https://img.shields.io/badge/domain-airway%20geometry-success" alt="Domain">
</p>

Procedural airway generation with **parametric primitives**, **smooth implicit junctions**, and **YAML-driven tree assembly** for simulation, morphometry, and controlled synthetic geometry studies.

</div>

---

## Overview

RespGeomLib is a lightweight Python research library for building **structured 3D airway lumen geometry** from compact, human-readable specifications.

Instead of relying on stitched tubular parts with visible seams at branch junctions, RespGeomLib combines:

- **analytic airway segments** for straight and tapered branches,
- **implicit junction modeling** for smooth bifurcations and trifurcations,
- **port-based assembly** for hierarchical tree construction,
- **YAML specifications** for reproducible geometry generation.

The goal is to make airway geometry generation more useful for:

- **CFD-ready geometry prototyping**
- **reproducible morphometry studies**
- **controlled stenosis/dilation experiments**
- **synthetic airway dataset generation**
- **educational and visualization workflows**

---

## Pipeline

<div align="center">
  <img src="pipeline-overview.png" alt="RespGeomLib pipeline" width="100%">
</div>

**Pipeline summary:** a compact YAML specification defines airway primitives and parent–child attachments through ports. Segments are placed via local-to-world frame transforms, junctions are generated using an implicit field, and the final surface is extracted and cleaned into an analysis-ready airway mesh.

---

## Why RespGeomLib?

CT-derived airway reconstruction is valuable, but it often comes with practical challenges such as segmentation artefacts, distal-resolution limits, and mesh cleanup around branch junctions.

RespGeomLib focuses on a **reproducible procedural alternative** with explicit control over geometry, branching, and local variation.

### Key advantages

- **Smooth branch junctions**  
  Avoids hard stitched seams by generating junctions through implicit surface construction.

- **Reproducible geometry**  
  The same YAML specification produces the same geometry workflow every time.

- **Modular tree assembly**  
  Each component exposes ports, making parent–child attachment clean and extensible.

- **Controlled pathology simulation**  
  Localized narrowing or dilation can be introduced in a structured way for comparative studies.

- **Research-friendly codebase**  
  Easy to inspect, modify, and extend for experiments in airway modeling, simulation, and synthetic generation.

---

## Core ideas

### 1. Parametric airway primitives
RespGeomLib uses reusable local-coordinate building blocks such as:

- **Pipe** — straight or tapered airway segments
- **Y2** — two-way branch junctions
- **Y3** — three-way branch junctions

### 2. Frame-based assembly
Every segment is defined in a local frame and attached to its parent through explicit ports. This makes it possible to build larger airway trees in a clean, deterministic way.

### 3. Implicit junction generation
Instead of directly stitching cylinders together, RespGeomLib constructs branch junctions using an implicit distance-field representation and extracts the surface locally.

### 4. YAML-driven specification
Airway trees can be defined in a compact YAML file with segment type, geometric parameters, parent ID, and parent port index.

---

## Repository structure

```text
RespGeomLib/
├── trees/
│   └── example_tree.yaml
├── example_Y_union.py
├── example_cylinder_vis.py
├── example_oriented_cylinder.py
├── example_y3_implicit_vis.py
├── example_y_implicit_vis.py
├── example_y_vis.py
├── example_y_vis_smooth.py
├── frames.py
├── implicit_y.py
├── implicit_y3.py
├── junctions.py
├── primitives.py
├── segments.py
├── tree_builder.py
├── tree_example.py
└── tree_example_y3.py
```

### Main modules

| File | Purpose |
|---|---|
| `frames.py` | Coordinate-frame utilities and direction construction from angles |
| `primitives.py` | Basic geometric primitives used by the pipeline |
| `segments.py` | Segment-level abstractions and port-aware geometry construction |
| `implicit_y.py` | Smooth implicit generation of two-way Y junctions |
| `implicit_y3.py` | Smooth implicit generation of three-way branching junctions |
| `junctions.py` | Junction-related helpers and mesh assembly logic |
| `tree_builder.py` | Builds airway trees from YAML specifications |
| `tree_example.py` / `tree_example_y3.py` | Example scripts for complete tree generation |
| `example_*` scripts | Small visualization and debugging examples |

---

## Installation

RespGeomLib is currently organized as a **research codebase** rather than a packaged Python library.

### Clone the repository

```bash
git clone https://github.com/nichula01/RespGeomLib.git
cd RespGeomLib
```

### Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install numpy pyvista pyyaml
```

---

## Quick start

### Build and visualize an example airway tree

```bash
python tree_builder.py
```

This loads:

```text
trees/example_tree.yaml
```

### Run example scripts

```bash
python tree_example.py
python tree_example_y3.py
python example_y_vis.py
python example_y_vis_smooth.py
python example_y_implicit_vis.py
python example_y3_implicit_vis.py
python example_cylinder_vis.py
python example_oriented_cylinder.py
python example_Y_union.py
```

---

## YAML tree specification

RespGeomLib supports tree generation from a YAML file where each segment entry includes:

- `id`
- `kind`
- `params`
- `parent_id`
- `parent_port_index`

### Example

```yaml
- id: root
  kind: pipe
  params:
    length: 6.0
    d_in: 2.0
    d_out: 2.0
  parent_id: null
  parent_port_index: null

- id: Y2_main
  kind: y2
  params:
    length_trunk: 6.0
    length_child1: 4.0
    length_child2: 4.0
    d_trunk: 2.0
    d_child1: 2.0
    d_child2: 2.0
    theta1_deg: 45.0
    phi1_deg: 0.0
    theta2_deg: 45.0
    phi2_deg: 120.0
  parent_id: root
  parent_port_index: 1
```

This structure makes the geometry definition compact, readable, and reproducible.

---

## Current capabilities

- Analytic pipe generation
- Tapered airway segments
- Two-way Y junction generation
- Three-way branching generation
- Port-aware hierarchical tree construction
- Local-to-world segment placement
- YAML-based airway tree definitions
- PyVista-based 3D visualization
- Smooth implicit branch blending
- Controlled geometry experimentation for airway studies

---

## Example research use cases

RespGeomLib can support workflows such as:

- **CFD pre-geometry generation** for airflow studies
- **airway morphometry experiments**
- **controlled pathology-inspired variants** such as stenosis and dilation
- **synthetic airway tree generation** for simulation or benchmarking
- **teaching and demonstration** of branching geometry pipelines
- **rapid prototyping** of airway structure models before larger simulation pipelines

---

## Project status

This repository is currently a **research-oriented prototype** under active development.

Planned improvements may include:

- `requirements.txt` or `pyproject.toml`
- export helpers for mesh formats such as STL, OBJ, or VTP
- better packaging and installation flow
- additional example YAML configurations
- pathology presets and more complex anatomical variants
- tests for frame transforms, ports, and tree assembly
- notebook-based demos

---

## Paper

The associated paper will be available on **arXiv soon**.

**Title:**  
*RespGeomLib: A Reproducible Parametric Engine for Generating Analysis-Ready Human Airway Lumen Geometry*

Until the preprint is public, this repository serves as the main project entry point.

> Once the arXiv link is live, add it here:
>
> ```md
> [Paper on arXiv](ARXIV_LINK_HERE)
> ```

---

## Citation

If you use this repository in academic work, please cite the paper when the final bibliographic details are available.

For now, you can use this placeholder:

```bibtex
@misc{wasalathilaka_respgeomlib,
  title  = {RespGeomLib: A Reproducible Parametric Engine for Generating Analysis-Ready Human Airway Lumen Geometry},
  author = {Nichula Sathmith Wasalathilaka and M. P. B. Ekanayake and G. M. R. I. Godaliyadda and Duminda Yasaratne},
  note   = {GitHub repository},
  year   = {2026}
}
```

---

## Contributing

Suggestions, issues, and improvements are welcome.

Good first contributions include:

- improving documentation,
- adding tests,
- improving YAML presets,
- adding mesh export utilities,
- extending branch-generation options,
- improving example scripts and visual outputs.

---

## Contact

**Nichula Sathmith Wasalathilaka**  
Department of Electrical and Electronic Engineering  
University of Peradeniya  
Sri Lanka

---

## Acknowledgment

RespGeomLib was developed as part of research on reproducible airway geometry generation for simulation-oriented and analysis-oriented respiratory modeling workflows.
