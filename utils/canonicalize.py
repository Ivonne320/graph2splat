import open3d as o3d 
import numpy as np

def align_gravity_with_plane(points_xyz):  # (N,3) world
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))
    # (plane) ax + by + cz + d = 0
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                             ransac_n=3, num_iterations=500)
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-8)
    z = np.array([0, 0, 1], dtype=np.float32)

    # rotation that maps n → z
    v = np.cross(n, z); s = np.linalg.norm(v); ccos = np.dot(n, z)
    if s < 1e-6:  # already aligned (or opposite)
        Rg = np.eye(3, dtype=np.float32) if ccos > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]], np.float32)
        Rg = np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1 - ccos) / (s**2))
    return Rg  # 3x3

def yaw_canonicalize_xy(points_g):  # after gravity align
    xy = points_g[:, :2]
    xy = xy - xy.mean(0)
    U, S, Vt = np.linalg.svd(xy, full_matrices=False)
    # first principal dir = Vt[0]; build Rz that maps it to +x
    main = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-8)
    theta = np.arctan2(main[1], main[0])
    c, s = np.cos(-theta), np.sin(-theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    return Rz