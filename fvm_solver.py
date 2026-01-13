import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

class DualMesh:
    """
    Handles the geometry of the primal triangular mesh and computes
    properties of the dual mesh (control volumes) required for Vertex-Centered FVM.
    """
    def __init__(self, points, simplices):
        """
        Args:
            points (np.ndarray): (N, 2) array of node coordinates.
            simplices (np.ndarray): (M, 3) array of triangle node indices.
        """
        self.points = points
        self.simplices = simplices
        self.n_points = points.shape[0]
        self.n_elements = simplices.shape[0]

        # Geometric properties to be computed
        self.control_volume_areas = np.zeros(self.n_points)
        
        # We will need to iterate elements to build these
        self._compute_geometry()

    def _compute_geometry(self):
        """
        Computes the area of dual cells (control volumes) associated with each node.
        In Vertex-Centered FVM on triangles, the control volume for a node
        is formed by connecting edge midpoints and the element centroid.
        This effectively assigns 1/3 of the triangle's area to each of its vertices.
        """
        for tri_indices in self.simplices:
            # Get coordinates of the triangle vertices
            pts = self.points[tri_indices] # Shape (3, 2)
            
            # Compute area of the triangle using cross product
            # Area = 0.5 * |(x1-x0)(y2-y0) - (x2-x0)(y1-y0)|
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = v1[0] * v2[1] - v1[1] * v2[0]
            tri_area = 0.5 * np.abs(cross_prod)
            
            # Add 1/3 of the area to each vertex's control volume
            for idx in tri_indices:
                self.control_volume_areas[idx] += tri_area / 3.0

class FVMSolver:
    """
    Solves the 2D Poisson equation -Laplacian(u) = f
    using Vertex-Centered Finite Volume Method.
    """
    def __init__(self, mesh: DualMesh):
        self.mesh = mesh
        self.A = None # Stiffness matrix
        self.b = None # Load vector
        self.u = np.zeros(mesh.n_points)

    def assemble(self, source_func):
        """
        Assembles the linear system A u = b.
        
        A: Global stiffness matrix (sparse).
        b: Load vector (integral of source term over control volumes).
        """
        n = self.mesh.n_points
        self.A = lil_matrix((n, n))
        self.b = np.zeros(n)
        
        points = self.mesh.points
        
        # --- Assembly Loop over Elements ---
        for tri_indices in self.mesh.simplices:
            # Vertices of the triangle
            # Local indices: 0, 1, 2
            # Global indices: tri_indices[0], tri_indices[1], tri_indices[2]
            local_pts = points[tri_indices] 
            
            # Compute geometric quantities for the element
            # We need gradients of shape functions (linear basis functions).
            # For linear triangle, u(x,y) = a + bx + cy
            # grad(u) = [b, c]
            # u = N0 u0 + N1 u1 + N2 u2
            # grad(u) = u0 grad(N0) + u1 grad(N1) + u2 grad(N2)
            
            # The contribution to the flux balance for a node i comes from
            # the integral of -grad(u) . n over the dual boundary inside this triangle.
            # Interestingly, for linear triangles, the FVM formulation 
            # turns out to be identical to P1 FEM stiffness matrix for the Laplacian term.
            # We will implement it using the "Element Stiffness Matrix" approach which is cleaner
            # and equivalent for this specific equation.
            
            # Area of triangle
            v1 = local_pts[1] - local_pts[0]
            v2 = local_pts[2] - local_pts[0]
            area = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            # Gradients of Basis Functions N_i
            # N_i = (a_i + b_i x + c_i y) / (2 * Area)
            # grad(N_i) = [b_i, c_i] / (2 * Area)
            # b_i = y_j - y_k, c_i = x_k - x_j (cyclic permutation)
            
            x = local_pts[:, 0]
            y = local_pts[:, 1]
            
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            # Element stiffness matrix K_local
            # K_ij = Integral(grad(N_i) . grad(N_j)) dA
            # Since grad(N) are constant:
            # K_ij = Area * (grad(N_i) . grad(N_j))
            #      = Area * (1/(2A) * [bi, ci]) . (1/(2A) * [bj, cj])
            #      = (1/(4A)) * (bi*bj + ci*cj)
            
            K_local = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    K_local[i, j] = (b[i]*b[j] + c[i]*c[j]) / (4.0 * area)
            
            # Add to global matrix
            for i in range(3):
                row = tri_indices[i]
                for j in range(3):
                    col = tri_indices[j]
                    self.A[row, col] += K_local[i, j]
            
            # --- Load Vector ---
            # Integral of f over Control Volume i.
            # Contribution from this triangle to CV_i is (Area/3) * f(centroid) OR f(node_i)
            # A simple approximation is f evaluated at the node * (Area/3)
            # Or f evaluated at centroid * (Area/3). We'll use node approximation (lumped).
            
            for i in range(3):
                idx = tri_indices[i]
                # Using 1/3 area approximation for contribution of this triangle to node's CV
                self.b[idx] += source_func(points[idx, 0], points[idx, 1]) * (area / 3.0)

    def apply_boundary_conditions(self, boundary_func):
        """
        Applies Dirichlet boundary conditions.
        Nodes on the boundary of the square [0,1]x[0,1] are fixed.
        """
        # Identify boundary nodes
        # For unit square, simple coordinate check
        tol = 1e-6
        points = self.mesh.points
        
        is_boundary = (
            (points[:, 0] < tol) | (points[:, 0] > 1.0 - tol) |
            (points[:, 1] < tol) | (points[:, 1] > 1.0 - tol)
        )
        
        # Modify A and b (Penalty method or Row-Zeroing w/ 1 on diag)
        # We use Row-Zeroing
        for i in range(self.mesh.n_points):
            if is_boundary[i]:
                # Zero out the row
                self.A[i, :] = 0.0
                # Set diagonal to 1
                self.A[i, i] = 1.0
                # Set RHS to boundary value
                x, y = points[i]
                self.b[i] = boundary_func(x, y)

    def solve(self):
        """
        Solves the linear system.
        """
        if self.A is None:
            raise RuntimeError("System not assembled. Call assemble() first.")
        
        # Convert to CSC for efficient solving
        A_csc = self.A.tocsc()
        self.u = spsolve(A_csc, self.b)
        return self.u
