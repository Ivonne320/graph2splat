import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def draw_wire_cube(ax, size=5):
    """Draw a size×size×size wire grid."""
    pts = np.arange(size)
    # edges along x, y, z
    for y in pts:
        for z in pts:
            ax.plot([0, size-1], [y, y], [z, z], color='k', lw=1.2, alpha=0.5)
    for x in pts:
        for z in pts:
            ax.plot([x, x], [0, size-1], [z, z], color='k', lw=1.2, alpha=0.5)
    for x in pts:
        for y in pts:
            ax.plot([x, x], [y, y], [0, size-1], color='k', lw=1.2, alpha=0.5)
    # corner dots
    # X, Y, Z = np.meshgrid(pts, pts, pts)
    # ax.scatter(X, Y, Z, color='k', s=8)

def draw_voxel(ax, center, color):
    """Add a semi-transparent cube centered at center."""
    x, y, z = center
    d = 0.4  # half-cube size
    corners = np.array([
        [x-d, y-d, z-d],
        [x+d, y-d, z-d],
        [x+d, y+d, z-d],
        [x-d, y+d, z-d],
        [x-d, y-d, z+d],
        [x+d, y-d, z+d],
        [x+d, y+d, z+d],
        [x-d, y+d, z+d],
    ])
    faces = [
        [corners[j] for j in [0,1,2,3]],
        [corners[j] for j in [4,5,6,7]],
        [corners[j] for j in [0,1,5,4]],
        [corners[j] for j in [2,3,7,6]],
        [corners[j] for j in [1,2,6,5]],
        [corners[j] for j in [4,7,3,0]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, edgecolor='k', linewidth=0.3, alpha=0.6)
    ax.add_collection3d(poly)

if __name__ == "__main__":
    size = 5
    # occupied = [(2,2,2), (3,3,3),(2,2,3),(3,2,2),(2,3,2),(1,2,2),(1,3,3),(3,1,2),(2,1,3),(3,2,3),(2,3,3),(1,1,2),(3,3,1),(1,1,1),(2,2,1)]
    occupied = []
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_wire_cube(ax, size=size)
    for cell in occupied:
        draw_voxel(ax, cell, color=(1.0, 0.4, 0.0))  # orange cubes
    ax.grid(False)
    # ax.set_frame_on(False)
    ax.set_axis_off()

    ax.set_xlim(-0.5, size-0.5)
    ax.set_ylim(-0.5, size-0.5)
    ax.set_zlim(-0.5, size-0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])
    ax.set_title("5×5×5 Occupancy Grid")
    plt.tight_layout()
    plt.savefig("./results/occupancy_grid_wireframe.png", dpi=200,transparent=True)
    