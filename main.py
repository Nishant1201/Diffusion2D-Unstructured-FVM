import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from utils import generate_unit_square_mesh, exact_solution, source_function_for_exact
from fvm_solver import DualMesh, FVMSolver

def main():
    # 1. Setup Parameters
    n_points = 41 # Points per side, so approx 400 nodes
    print(f"Generating mesh with {n_points}x{n_points} points...")
    
    # 2. Generate Mesh
    points, simplices = generate_unit_square_mesh(n_points)
    
    # 3. Initialize Solver components
    mesh = DualMesh(points, simplices)
    solver = FVMSolver(mesh)
    
    # 4. Assemble System
    print("Assembling system...")
    solver.assemble(source_function_for_exact)
    
    # 5. Apply Boundary Conditions
    # For exact solution u = sin(pi*x)*sin(pi*y), BC is u=0 on boundary.
    print("Applying Boundary Conditions...")
    def bc_func(x, y):
        return exact_solution(x, y) # Should be 0 on boundary
        
    solver.apply_boundary_conditions(bc_func)
    
    # 6. Solve
    print("Solving linear system...")
    u_hm = solver.solve()
    
    # 7. Compute Error
    u_exact = exact_solution(points[:, 0], points[:, 1])
    error = np.abs(u_hm - u_exact)
    l2_error = np.sqrt(np.sum(error**2) / len(points))
    print(f"L2 Error: {l2_error:.6e}")
    
    # 8. Visualization
    print("Visualizing results...")
    
    # Create triangulation for plotting
    triang = mtri.Triangulation(points[:, 0], points[:, 1], simplices)
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # Plot Mesh (New)
    axes[0].triplot(triang, 'k-', linewidth=0.5)
    axes[0].set_title("Mesh")
    axes[0].set_aspect('equal')

    # Plot Computed Solution
    tcf0 = axes[1].tricontourf(triang, u_hm, levels=14, cmap="jet")
    fig.colorbar(tcf0, ax=axes[1])
    axes[1].set_title("Computed Solution (FVM)")
    axes[1].set_aspect('equal')
    
    # Plot Exact Solution
    tcf1 = axes[2].tricontourf(triang, u_exact, levels=14, cmap="jet")
    fig.colorbar(tcf1, ax=axes[2])
    axes[2].set_title("Exact Solution")
    axes[2].set_aspect('equal')
    
    # Plot Error
    tcf2 = axes[3].tricontourf(triang, error, levels=14, cmap="jet")
    fig.colorbar(tcf2, ax=axes[3])
    axes[3].set_title("Absolute Error")
    axes[3].set_aspect('equal')
    
    plt.tight_layout()
    output_file = 'result_fvm.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
