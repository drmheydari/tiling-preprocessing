import numpy as np
import plotly.graph_objects as go
import numexpr as ne  # Import NumExpr for optimized computations
from concurrent.futures import ThreadPoolExecutor

# Parameters
N = 100000  # Number of random points
range_x = (0, 200)
range_y = (0, 200)
range_z = (0, 200)

# Generate random points
x = np.random.uniform(range_x[0], range_x[1], N)
y = np.random.uniform(range_y[0], range_y[1], N)
z = np.random.uniform(range_z[0], range_z[1], N)

# Define colors for each recursive level
colors = ["blue", "red", "green", "purple", "orange", "pink", "cyan", "yellow", "brown", "magenta"]

# **Precompute Extreme Points**
def get_extreme_points(x, y, z, corner):
    if corner == "top-left-near":
        return np.argmin(x), np.argmax(y), np.argmin(z)
    elif corner == "top-left-far":
        return np.argmin(x), np.argmax(y), np.argmax(z)
    elif corner == "top-right-near":
        return np.argmax(x), np.argmax(y), np.argmin(z)
    elif corner == "top-right-far":
        return np.argmax(x), np.argmax(y), np.argmax(z)
    elif corner == "bottom-left-near":
        return np.argmin(x), np.argmin(y), np.argmin(z)
    elif corner == "bottom-left-far":
        return np.argmin(x), np.argmin(y), np.argmax(z)
    elif corner == "bottom-right-near":
        return np.argmax(x), np.argmin(y), np.argmin(z)
    elif corner == "bottom-right-far":
        return np.argmax(x), np.argmin(y), np.argmax(z)
    else:
        raise ValueError("Invalid corner selection.")

# **Recursive Function to Find Bounding Cuboid (Optimized)**
def recursive_bounding_cuboid(x, y, z, fig, corner, depth=0, draw_cuboid=True):
    if len(x) <= 3 or depth >= len(colors):
        return  # Stop recursion if at most 3 points are left or colors are exhausted

    # Get extreme points
    extreme_x_idx, extreme_y_idx, extreme_z_idx = get_extreme_points(x, y, z, corner)

    # Define the bounding cuboid using NumPy (using np.amin and np.amax for optimization)
    x_vals = np.array([x[extreme_x_idx], x[extreme_y_idx], x[extreme_z_idx]])
    y_vals = np.array([y[extreme_x_idx], y[extreme_y_idx], y[extreme_z_idx]])
    z_vals = np.array([z[extreme_x_idx], z[extreme_y_idx], z[extreme_z_idx]])

    x_min, x_max = np.amin(x_vals), np.amax(x_vals)
    y_min, y_max = np.amin(y_vals), np.amax(y_vals)
    z_min, z_max = np.amin(z_vals), np.amax(z_vals)

    # **Optimized NumExpr Calculation for Filtering Points**
    inside_cuboid = ne.evaluate(
        "(x > x_min) & (x < x_max) & (y > y_min) & (y < y_max) & (z > z_min) & (z < z_max)",
        local_dict={'x': x, 'y': y, 'z': z, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'z_min': z_min, 'z_max': z_max}
    )

    # Extract points inside the cuboid
    inside_x = x[inside_cuboid]
    inside_y = y[inside_cuboid]
    inside_z = z[inside_cuboid]

    # **Draw the bounding cuboid (only for the first level)**
    if draw_cuboid:
        # Define the vertices of the cuboid
        cuboid_vertices = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ]

        # Define the edges of the cuboid
        cuboid_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]

        # Plot the cuboid edges
        for edge in cuboid_edges:
            fig.add_trace(go.Scatter3d(
                x=[cuboid_vertices[edge[0]][0], cuboid_vertices[edge[1]][0]],
                y=[cuboid_vertices[edge[0]][1], cuboid_vertices[edge[1]][1]],
                z=[cuboid_vertices[edge[0]][2], cuboid_vertices[edge[1]][2]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

    # **Color all points inside the cuboid uniformly**
    fig.add_trace(go.Scatter3d(
        x=inside_x,
        y=inside_y,
        z=inside_z,
        mode='markers',
        marker=dict(size=3, color=colors[depth]),  # Original point size (size=3)
        showlegend=False
    ))

    # **Recursive call for the points inside the cuboid**
    recursive_bounding_cuboid(inside_x, inside_y, inside_z, fig, corner, depth + 1, draw_cuboid=False)


# Parallelize the recursive function calls for different corners
def run_parallel_recursion(fig, x, y, z):
    # Define all corners to process
    corners = [
        "top-left-near", "top-left-far",
        "top-right-near", "top-right-far",
        "bottom-left-near", "bottom-left-far",
        "bottom-right-near", "bottom-right-far"
    ]

    # Run the recursive function for all corners in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(recursive_bounding_cuboid, x, y, z, fig, corner) for corner in corners]
        for future in futures:
            future.result()


# Initialize figure
fig = go.Figure()

# Plot the original points (black)
fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(size=2, color='black'),  # Original point size (size=2)
    showlegend=False
))

# Run the recursive function in parallel for all corners
run_parallel_recursion(fig, x, y, z)

# Layout settings
fig.update_layout(
    title="Tiling in 3D",
    scene=dict(
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        zaxis_title="Z-axis",
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    showlegend=False
)

# Show the plot
fig.show()
