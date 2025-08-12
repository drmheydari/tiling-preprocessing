import numpy as np
import numexpr as ne  # Import NumExpr for optimized computations
from pyhull.convex_hull import ConvexHull  # Import pyhull

# # Generate square-shaped points (randomly inside a square)
# N = 100
# square_min = 0
# square_max = 200

# np.random.seed(42)
# x_vals = np.random.uniform(square_min, square_max, N)
# y_vals = np.random.uniform(square_min, square_max, N)

# Parameters
N = 50000  # Number of random points
range_x = (0, 200)
range_y = (0, 200)

# Generate random points
x_vals = np.random.uniform(range_x[0], range_x[1], N)
y_vals = np.random.uniform(range_y[0], range_y[1], N)

# Set to store extreme points (Change 2: using set to avoid duplicates)
extreme_points = set()


# **Generalized Recursive Function to Store Extreme Points**
def recursive_bounding_box(x_vals, y_vals, corner="top-left", depth=0):
    # Change 1: If stopping condition is met, add all points to the extreme set
    if len(x_vals) <= 3 or depth >= 10:
        for i in range(len(x_vals)):
            extreme_points.add((float(x_vals[i]), float(y_vals[i])))
        return

    # **Determine extreme points based on the selected corner**
    if corner == "top-left":
        extreme_x_indices = np.where(x_vals == np.min(x_vals))[0]  # all leftmost points
        extreme_y_indices = np.where(y_vals == np.max(y_vals))[0]  # all topmost points
    elif corner == "top-right":
        extreme_x_indices = np.where(x_vals == np.max(x_vals))[0]  # all rightmost points
        extreme_y_indices = np.where(y_vals == np.max(y_vals))[0]  # all topmost points
    elif corner == "right-bottom":
        extreme_x_indices = np.where(x_vals == np.max(x_vals))[0]  # all rightmost points
        extreme_y_indices = np.where(y_vals == np.min(y_vals))[0]  # all bottommost points
    elif corner == "bottom-left":
        extreme_x_indices = np.where(x_vals == np.min(x_vals))[0]  # all leftmost points
        extreme_y_indices = np.where(y_vals == np.min(y_vals))[0]  # all bottommost points
    else:
        raise ValueError("Invalid corner selection. Use 'top-left', 'top-right', 'right-bottom', or 'bottom-left'.")

    # Change 3: Add all extreme points to the set
    for idx in extreme_x_indices:
        extreme_points.add((float(x_vals[idx]), float(y_vals[idx])))
    for idx in extreme_y_indices:
        extreme_points.add((float(x_vals[idx]), float(y_vals[idx])))

    # For continuing recursion, use the last point
    extreme_x_idx = extreme_x_indices[-1]
    extreme_y_idx = extreme_y_indices[-1]

    # **Define bounding box based on extremes**
    x_min, x_max = min(x_vals[extreme_x_idx], x_vals[extreme_y_idx]), max(x_vals[extreme_x_idx], x_vals[extreme_y_idx])
    y_min, y_max = min(y_vals[extreme_x_idx], y_vals[extreme_y_idx]), max(y_vals[extreme_x_idx], y_vals[extreme_y_idx])

    # **Optimized NumExpr Calculation for Filtering Points**
    inside_box = ne.evaluate("(x_vals > x_min) & (x_vals < x_max) & (y_vals > y_min) & (y_vals < y_max)",
                             local_dict={'x_vals': x_vals, 'y_vals': y_vals, 'x_min': x_min, 'x_max': x_max,
                                         'y_min': y_min, 'y_max': y_max})

    # **Check if the box actually shrinks**
    if np.count_nonzero(inside_box) >= len(x_vals):
        return

    # Extract filtered points
    inside_x = x_vals[inside_box]
    inside_y = y_vals[inside_box]

    # **Recursive call to store extreme points (without visualization)**
    recursive_bounding_box(inside_x, inside_y, corner, depth + 1)


# **Run the function for each corner and store extreme points**
recursive_bounding_box(x_vals, y_vals, "top-left")
recursive_bounding_box(x_vals, y_vals, "top-right")
recursive_bounding_box(x_vals, y_vals, "right-bottom")
recursive_bounding_box(x_vals, y_vals, "bottom-left")

# Convert set to numpy array for pyhull
extreme_points_array = np.array(list(extreme_points))

# Create initial points as numpy array
original_points = np.column_stack((x_vals, y_vals))

print("Number of original points:", len(original_points))
print("Number of extreme points after preprocessing:", len(extreme_points_array))

# Compute convex hull for original points
try:
    hull_original = ConvexHull(original_points.tolist())
    vertices_original = hull_original.vertices
    print("Original hull calculated successfully")
except Exception as e:
    print("Error with original points:", e)
    vertices_original = []

# Compute convex hull for preprocessed points
try:
    hull_preprocessed = ConvexHull(extreme_points_array.tolist())
    vertices_preprocessed = hull_preprocessed.vertices
    print("Preprocessed hull calculated successfully")
except Exception as e:
    print("Error with preprocessed points:", e)
    vertices_preprocessed = []

print("\n=== Convex Hull Results ===")
print("Original points convex hull vertices count:", len(vertices_original))
print("Preprocessed points convex hull vertices count:", len(vertices_preprocessed))

if len(vertices_original) > 0:
    print("\nConvex hull vertices from original points:")
    hull_points_original = original_points[np.array(vertices_original).flatten()]
    for i, point in enumerate(hull_points_original):
        x = float(point[0])
        y = float(point[1])
        print(f"  Vertex {i}: ({x:.4f}, {y:.4f})")

if len(vertices_preprocessed) > 0:
    print("\nConvex hull vertices from preprocessed points:")
    hull_points_preprocessed = extreme_points_array[np.array(vertices_preprocessed).flatten()]
    for i, point in enumerate(hull_points_preprocessed):
        x = float(point[0])
        y = float(point[1])
        print(f"  Vertex {i}: ({x:.4f}, {y:.4f})")


if len(vertices_original) == 0 or len(vertices_preprocessed) == 0:
    print("Could not compute convex hull properly.")
