import numpy as np


def ray_sphere_intersection_vectorized(ray_origins, ray_dirs, sphere_center, sphere_radius):
    """
    完全矢量化的射线-球面交点计算
    """
    # 向量从球心到射线起点
    oc = ray_origins - sphere_center

    # 使用更快的点积计算
    a = (ray_dirs * ray_dirs).sum(axis=1)
    b = 2 * (oc * ray_dirs).sum(axis=1)
    c = (oc * oc).sum(axis=1) - sphere_radius**2

    # 判别式
    discriminant = b * b - 4 * a * c

    # 预分配结果
    intersections = np.zeros_like(ray_origins)
    final_valid = np.zeros(len(ray_origins), dtype=bool)

    # 有解的掩码
    valid_mask = discriminant >= 0

    if not np.any(valid_mask):
        return intersections, final_valid

    # 直接在原数组上操作，减少临时数组
    sqrt_disc = np.zeros_like(discriminant)
    sqrt_disc[valid_mask] = np.sqrt(discriminant[valid_mask])

    # 计算t值
    inv_2a = np.zeros_like(a)
    inv_2a[valid_mask] = 1.0 / (2 * a[valid_mask])

    t1 = (-b - sqrt_disc) * inv_2a
    t2 = (-b + sqrt_disc) * inv_2a

    # 选择有效的t
    eps = 1e-6
    t = np.where(t1 > eps, t1, t2)

    # 最终掩码
    final_valid = (valid_mask) & (t > eps)

    # 计算交点
    intersections[final_valid] = (ray_origins[final_valid] +
                                  t[final_valid, np.newaxis] * ray_dirs[final_valid])

    return intersections, final_valid

def compute_refraction_vectorized(incident_dirs, normals, n1, n2):
    """
    完全矢量化的折射计算
    """
    # 计算入射角余弦
    cos_i = -np.sum(incident_dirs * normals, axis=1)

    # 处理从内部射向外部的情况
    flip_mask = cos_i < 0
    cos_i = np.abs(cos_i)

    # 计算折射率比
    eta = n1 / n2

    # 计算判别式
    sin_i_sq = 1 - cos_i**2
    sin_t_sq = eta**2 * sin_i_sq

    # 全反射判断
    total_reflection = sin_t_sq > 1.0

    # 计算折射方向
    cos_t = np.sqrt(np.maximum(0, 1 - sin_t_sq))

    # 计算折射方向向量
    refract_dirs = np.zeros_like(incident_dirs)
    valid_mask = ~total_reflection

    if np.any(valid_mask):
        eta_cos_i = eta * cos_i[valid_mask]
        factor = (eta_cos_i - cos_t[valid_mask])[:, np.newaxis]
        refract_dirs[valid_mask] = (eta * incident_dirs[valid_mask] -
                                    factor * normals[valid_mask])

        # 归一化
        norms = np.linalg.norm(refract_dirs[valid_mask], axis=1)
        refract_dirs[valid_mask] = refract_dirs[valid_mask] / norms[:, np.newaxis]

    return refract_dirs, total_reflection

def is_front_surface_improved(pmt_pos, intersection_points):
    """
    改进的PMT前表面判断
    """
    pmt_center = pmt_pos
    pmt_to_center_dir = -pmt_center / np.linalg.norm(pmt_center)

    # 计算交点相对于PMT中心的方向
    relative_pos = intersection_points - pmt_center
    relative_dirs = relative_pos / np.linalg.norm(relative_pos, axis=1)[:, np.newaxis]

    # 前表面：法向量与指向球心方向的夹角 < 90度
    dot_products = np.sum(relative_dirs * pmt_to_center_dir, axis=1)
    return dot_products > 0
