import numpy as np
import plotly.graph_objects as go
import numexpr as ne  # Import NumExpr for optimized computations

# Parameters
N = 100000  # Number of random points
range_x = (0, 200)
range_y = (0, 200)

# Generate random points
x_vals = np.random.uniform(range_x[0], range_x[1], N)
y_vals = np.random.uniform(range_y[0], range_y[1], N)

# Define colors for each recursive level
colors = ["red", "blue", "green", "purple", "orange", "pink", "cyan", "yellow", "brown", "magenta"]


# ** Optimized Recursive Function with Bounding Box Drawing **
def recursive_bounding_box(x_vals, y_vals, fig, corner="top-left", depth=0, draw_box=True):
    if len(x_vals) <= 3 or depth >= len(colors):
        return  # Stop recursion if at most 3 points are left or colors are exhausted

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
    x_min, x_max = min(x_vals[extreme_x_idx], x_vals[extreme_y_idx]), max(x_vals[extreme_x_idx], x_vals[extreme_y_idx])
    y_min, y_max = min(y_vals[extreme_x_idx], y_vals[extreme_y_idx]), max(y_vals[extreme_x_idx], y_vals[extreme_y_idx])

    # **Optimized NumExpr Calculation for Filtering Points**
    inside_box = ne.evaluate("(x_vals > x_min) & (x_vals < x_max) & (y_vals > y_min) & (y_vals < y_max)",
                             local_dict={'x_vals': x_vals, 'y_vals': y_vals, 'x_min': x_min, 'x_max': x_max,
                                         'y_min': y_min, 'y_max': y_max})

    if np.count_nonzero(inside_box) >= len(x_vals):  # Ensure bounding box shrinks
        return

    # Extract filtered points
    inside_x = x_vals[inside_box]
    inside_y = y_vals[inside_box]

    # **First Level Only: Draw Bounding Box (No Caption)**
    if draw_box:
        fig.add_trace(go.Scatter(
            x=[x_min, x_max, x_max, x_min, x_min],  # Closing the rectangle
            y=[y_min, y_min, y_max, y_max, y_min],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False  # Hides caption
        ))

    # Plot the points inside the bounding box with a different color
    fig.add_trace(go.Scatter(
        x=inside_x,
        y=inside_y,
        mode='markers',
        marker=dict(size=2, color=colors[depth]),  # Smaller points
        showlegend=False  # Hides caption
    ))

    # **Optimized Recursion without drawing bounding boxes again**
    recursive_bounding_box(inside_x, inside_y, fig, corner, depth + 1, draw_box=False)


# Initialize figure
fig = go.Figure()

# Plot the original points (Smaller Size, No Caption)
fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='markers',
    marker=dict(size=1.5, color='black'),
    showlegend=False
))

# **Run the function 4 times for each corner**
recursive_bounding_box(x_vals, y_vals, fig, "top-left")
recursive_bounding_box(x_vals, y_vals, fig, "top-right")
recursive_bounding_box(x_vals, y_vals, fig, "right-bottom")
recursive_bounding_box(x_vals, y_vals, fig, "bottom-left")

# Layout settings
fig.update_layout(
    title="Recursive Bounding Box for All Four Corners (Optimized)",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    margin=dict(l=0, r=0, b=0, t=50),
    showlegend=False  # Hide all captions
)

# Show the plot
fig.show()
