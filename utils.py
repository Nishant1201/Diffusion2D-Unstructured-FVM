import numpy as np
from scipy.spatial import Delaunay

def generate_unit_square_mesh(n_points_per_side):
    """
    Generates a Delaunay triangulation of the unit square [0,1]x[0,1].
    
    Args:
        n_points_per_side (int): Number of points along one side of the square.
        
    Returns:
        points (np.ndarray): (N, 2) array of node coordinates.
        simplices (np.ndarray): (M, 3) array of triangle node indices.
    """
    x = np.linspace(0, 1, n_points_per_side)
    y = np.linspace(0, 1, n_points_per_side)
    points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    
    # Perturb internal points slightly to make it "unstructured-like" if desired,
    # but for basic testing, a structured grid triangulated is fine.
    # We stick to a regular grid triangulation for predictability in initial tests.
    
    tri = Delaunay(points)
    
    return points, tri.simplices

def exact_solution(x, y):
    """
    Exact solution for validation: u(x,y) = sin(pi*x) * sin(pi*y)
    Source term f should be: 2*pi^2 * sin(pi*x) * sin(pi*y)
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def source_function_for_exact(x, y):
    """
    Source term corresponding to the exact solution above.
    """
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
