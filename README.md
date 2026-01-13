# Diffusion2D Unstructured Solver

A Python-based 2D diffusion equation solver using the Vertex-Centered Finite Volume Method (FVM) on unstructured triangular meshes.

## Overview

This project implements a solver for the Poisson equation:
$$ -\nabla^2 u = f $$
on a unit square domain $[0,1] \times [0,1]$. It uses an unstructured grid generated via Delaunay triangulation and solves the system using a Vertex-Centered Finite Volume scheme. The results are verified against an analytical exact solution.

## Features

- **Unstructured Mesh**: Generates triangular meshes using `scipy.spatial.Delaunay`.
- **Finite Volume Method**: Implements a vertex-centered FVM suitable for triangular elements.
- **Boundary Conditions**: Applies Dirichlet boundary conditions (fixed values on boundaries).
- **Error Analysis**: Computes L2 error norm against the exact analytical solution.
- **Visualization**: Plots the Mesh, Computed Solution, Exact Solution, and Absolute Error using `matplotlib`.

## Installation

Ensure you have Python 3.8+ installed. Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to generate the mesh, solve the system, and visualize the results:

```bash
python main.py
```

## Results

After running the script, a plot window will appear showing:
1.   **Mesh**: The generated unstructured triangular mesh.
2.   **Computed Solution**: The numerical solution $u_{hm}$.
3.   **Exact Solution**: The analytical reference solution.
4.   **Absolute Error**: The difference $|u_{hm} - u_{exact}|$.

The figure is also saved as `result_fvm.png`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
