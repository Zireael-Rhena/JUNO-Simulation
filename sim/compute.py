import numpy as np

def ray_sphere_intersection_vectorized(ray_origins, ray_dirs, sphere_center, sphere_radius):
    """
    射线-球面交点计算。
    
    使用矢量化操作计算多条射线与球面的交点，提高计算效率。
    采用射线-球面求交的标准算法，通过求解二次方程来确定交点。
    
    Args:
        ray_origins (numpy.ndarray): 射线起点坐标，形状为 (N, 3)
        ray_dirs (numpy.ndarray): 射线方向向量（应已归一化），形状为 (N, 3)
        sphere_center (numpy.ndarray): 球心坐标，形状为 (3,)
        sphere_radius (float): 球面半径
        
    Returns:
        tuple: 包含两个元素的元组
            - intersections (numpy.ndarray): 交点坐标，形状为 (N, 3)
            - final_valid (numpy.ndarray): 布尔掩码，标识有效交点，形状为 (N,)
            
    Note:
        - 对于每条射线，优先选择较近的交点（t1）
        - 如果t1 <= 0，则选择较远的交点（t2）
        - 使用eps=1e-6作为数值精度阈值
    """
    # 向量从球心到射线起点
    oc = ray_origins - sphere_center

    # 使用点积计算
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
    光线折射计算。
    
    根据斯涅尔定律计算光线在介质界面的折射方向。使用矢量化操作
    同时处理多条光线，提高计算效率。自动处理全反射情况。
    
    Args:
        incident_dirs (numpy.ndarray): 入射光线方向向量，形状为 (N, 3)
        normals (numpy.ndarray): 界面法向量（指向入射介质），形状为 (N, 3)
        n1 (float): 入射介质折射率
        n2 (float): 折射介质折射率
        
    Returns:
        tuple: 包含两个元素的元组
            - refract_dirs (numpy.ndarray): 折射光线方向向量，形状为 (N, 3)
            - total_reflection (numpy.ndarray): 全反射掩码，形状为 (N,)，
              True表示发生全反射，False表示发生折射
              
    Note:
        - 使用斯涅尔定律: n1 * sin(θ1) = n2 * sin(θ2)
        - 当 sin²(θ2) > 1 时发生全反射
        - 自动处理光线从内部射向外部的情况（cos_i < 0）
        - 折射方向向量已归一化
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
    PMT前表面判断算法。
    
    判断光线与PMT球体的交点是否位于前表面（面向探测器中心的一侧）。
    通过计算交点相对于PMT中心的位置向量与PMT指向球心方向的夹角来判断。
    
    Args:
        pmt_pos (numpy.ndarray): PMT中心位置坐标，形状为 (3,)
        intersection_points (numpy.ndarray): 光线与PMT的交点坐标，形状为 (N, 3)
        
    Returns:
        numpy.ndarray: 布尔数组，形状为 (N,)
            True表示交点位于前表面，False表示位于后表面
            
    Note:
        - 前表面定义：法向量与指向球心方向的夹角 < 90度（点积 > 0）
        - 后表面定义：法向量与指向球心方向的夹角 > 90度（点积 < 0）
        - 假设PMT位于以原点为中心的球面上
    """
    pmt_center = pmt_pos
    pmt_to_center_dir = -pmt_center / np.linalg.norm(pmt_center)

    # 计算交点相对于PMT中心的方向
    relative_pos = intersection_points - pmt_center
    relative_dirs = relative_pos / np.linalg.norm(relative_pos, axis=1)[:, np.newaxis]

    # 前表面：法向量与指向球心方向的夹角 < 90度
    dot_products = np.sum(relative_dirs * pmt_to_center_dir, axis=1)
    return dot_products > 0
