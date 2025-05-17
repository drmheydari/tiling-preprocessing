import numpy as np
import numexpr as ne
import time  # For timing the execution

# Parameters
N = 1000000  # Number of random points
range_x = (0, 200)
range_y = (0, 200)
range_z = (0, 200)

# Generate random points
x = np.random.uniform(range_x[0], range_x[1], N)
y = np.random.uniform(range_y[0], range_y[1], N)
z = np.random.uniform(range_z[0], range_z[1], N)

# List to store all extreme points
extreme_points = []


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


# **Recursive Function to Find Bounding Cuboid**
def recursive_bounding_cuboid(x, y, z, corner, depth=0):
    if len(x) <= 3 or depth >= 10:  # Stop recursion if at most 3 points are left or depth exceeds
        return  # Terminate the recursion

    # Get extreme points
    extreme_x_idx, extreme_y_idx, extreme_z_idx = get_extreme_points(x, y, z, corner)

    # Store extreme points as Python floats
    extreme_points.append(
        [float(x[extreme_x_idx]), float(y[extreme_x_idx]), float(z[extreme_x_idx])])  # Extreme x point
    extreme_points.append(
        [float(x[extreme_y_idx]), float(y[extreme_y_idx]), float(z[extreme_y_idx])])  # Extreme y point
    extreme_points.append(
        [float(x[extreme_z_idx]), float(y[extreme_z_idx]), float(z[extreme_z_idx])])  # Extreme z point

    # Define the bounding cuboid using NumPy
    x_min = min(x[extreme_x_idx], x[extreme_y_idx], x[extreme_z_idx])
    x_max = max(x[extreme_x_idx], x[extreme_y_idx], x[extreme_z_idx])
    y_min = min(y[extreme_x_idx], y[extreme_y_idx], y[extreme_z_idx])
    y_max = max(y[extreme_x_idx], y[extreme_y_idx], y[extreme_z_idx])
    z_min = min(z[extreme_x_idx], z[extreme_y_idx], z[extreme_z_idx])
    z_max = max(z[extreme_x_idx], z[extreme_y_idx], z[extreme_z_idx])

    # **Optimized NumExpr Calculation for Filtering Points**
    inside_cuboid = ne.evaluate(
        "(x > x_min) & (x < x_max) & (y > y_min) & (y < y_max) & (z > z_min) & (z < z_max)",
        local_dict={'x': x, 'y': y, 'z': z, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
                    'z_min': z_min, 'z_max': z_max}
    )

    # Extract points inside the cuboid
    inside_x = x[inside_cuboid]
    inside_y = y[inside_cuboid]
    inside_z = z[inside_cuboid]

    # **Recursive call for the points inside the cuboid**
    recursive_bounding_cuboid(inside_x, inside_y, inside_z, corner, depth + 1)


# Start timing
start = time.perf_counter()

# **Run the recursive function for all eight corners**
corners = [
    "top-left-near", "top-left-far",
    "top-right-near", "top-right-far",
    "bottom-left-near", "bottom-left-far",
    "bottom-right-near", "bottom-right-far"
]

for corner in corners:
    recursive_bounding_cuboid(x, y, z, corner)

# Calculate the time taken
end = time.perf_counter()  # End high-resolution timer

# Calculate the total time taken by all recursive calls
execution_time = end - start

# Print all extreme points
print("Extreme Points Collected:")
for point in extreme_points:
    print(point)

# Print execution time
print(f"Execution Time: {execution_time:.4f} seconds")
