import numpy as np
import numexpr as ne  # Import NumExpr for optimized computations
import time  # For higher resolution timing

# Parameters
N = 1000000  # Number of random points
range_x = (0, 200)
range_y = (0, 200)

# Generate random points
x_vals = np.random.uniform(range_x[0], range_x[1], N)
y_vals = np.random.uniform(range_y[0], range_y[1], N)

# List to store extreme points
extreme_points = []



# **Generalized Recursive Function to Store Extreme Points**
def recursive_bounding_box(x_vals, y_vals, corner="top-left", depth=0):
    if len(x_vals) <= 3 or depth >= 10:  # Stop recursion if at most 3 points are left or depth exceeds
        return  # Terminate the recursion

    # **Determine extreme points based on the selected corner**
    if corner == "top-left":
        extreme_x_idx = np.argmin(x_vals)  # Leftmost
        extreme_y_idx = np.argmax(y_vals)  # Topmost
    elif corner == "top-right":
        extreme_x_idx = np.argmax(x_vals)  # Rightmost
        extreme_y_idx = np.argmax(y_vals)  # Topmost
    elif corner == "right-bottom":
        extreme_x_idx = np.argmax(x_vals)  # Rightmost
        extreme_y_idx = np.argmin(y_vals)  # Bottommost
    elif corner == "bottom-left":
        extreme_x_idx = np.argmin(x_vals)  # Leftmost
        extreme_y_idx = np.argmin(y_vals)  # Bottommost
    else:
        raise ValueError("Invalid corner selection. Use 'top-left', 'top-right', 'right-bottom', or 'bottom-left'.")

    # Get extreme points directly
    extreme_x_point = [x_vals[extreme_x_idx], y_vals[extreme_x_idx]]
    extreme_y_point = [x_vals[extreme_y_idx], y_vals[extreme_y_idx]]

    # Store extreme points in the list
    extreme_points.append([float(extreme_x_point[0]), float(extreme_x_point[1])])  # Convert to float
    extreme_points.append([float(extreme_y_point[0]), float(extreme_y_point[1])])  # Convert to float

    # **Define bounding box based on extremes**
    x_min, x_max = min(x_vals[extreme_x_idx], x_vals[extreme_y_idx]), max(x_vals[extreme_x_idx], x_vals[extreme_y_idx])
    y_min, y_max = min(y_vals[extreme_x_idx], y_vals[extreme_y_idx]), max(y_vals[extreme_x_idx], y_vals[extreme_y_idx])

    # **Optimized NumExpr Calculation for Filtering Points**
    inside_box = ne.evaluate("(x_vals > x_min) & (x_vals < x_max) & (y_vals > y_min) & (y_vals < y_max)",
                             local_dict={'x_vals': x_vals, 'y_vals': y_vals, 'x_min': x_min, 'x_max': x_max,
                                         'y_min': y_min, 'y_max': y_max})

    # **Check if the box shrinks**
    if np.count_nonzero(inside_box) >= len(x_vals):
        return

    # Extract filtered points
    inside_x = x_vals[inside_box]
    inside_y = y_vals[inside_box]

    # **Optimized Recursion to store extreme points (without drawing anything)**
    recursive_bounding_box(inside_x, inside_y, corner, depth + 1)


# Measure the time taken by the recursive calls
start = time.perf_counter()  # Start high-resolution timer

# **Run the function for each corner and store extreme points**
recursive_bounding_box(x_vals, y_vals, "top-left")
recursive_bounding_box(x_vals, y_vals, "top-right")
recursive_bounding_box(x_vals, y_vals, "right-bottom")
recursive_bounding_box(x_vals, y_vals, "bottom-left")

# Calculate the time taken
end = time.perf_counter()  # End high-resolution timer

# Calculate the total time taken by all recursive calls
execution_time = end - start

# Now, extreme_points contains the stored extreme points
print("Extreme points collected:", extreme_points)
print("Length Extreme points collected:", len(extreme_points))
print(f"Time taken for execution: {execution_time:.4f} seconds")
