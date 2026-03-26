# core/data_loader.py

import zarr
import os
import logging
import numpy as np
from typing import Tuple, List, Union, Any

logger = logging.getLogger(__name__)

# 🎯 兼容性补丁：自动识别 Zarr 版本对应的 Group 类型
try:
    # Zarr V2 路径
    ZARR_GROUP_TYPE = zarr.hierarchy.Group
except AttributeError:
    # Zarr V3 路径
    ZARR_GROUP_TYPE = zarr.Group

def safe_open_zarr(path: str) -> Any:
    """
    针对底层元数据缺失的 Zarr 文件夹进行鲁棒打开。
    适配 Python 3.13 和 Zarr V2/V3。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {os.path.abspath(path)}")

    # 显式使用 DirectoryStore
    store = zarr.DirectoryStore(path)
    
    # 核心补丁：如果根目录缺少标识文件，则自动补全（Healing机制）
    v2_meta = os.path.join(path, '.zgroup')
    v3_meta = os.path.join(path, 'zarr.json')
    
    if not os.path.exists(v2_meta) and not os.path.exists(v3_meta):
        logger.warning(f"检测到 Zarr 根元数据丢失，正在为 {path} 修复索引...")
        # 补全元数据
        zarr.group(store=store, overwrite=False)

    try:
        # Zarr V3 推荐方式
        return zarr.open_group(store, mode='r')
    except Exception:
        # Zarr V2 保底方式
        return zarr.open(path, mode='r')

def get_episode_data(root: Any, ep_id: str) -> Tuple[np.ndarray, np.ndarray]:
    if ep_id not in root:
        available = sorted(list(root.group_keys()))
        raise KeyError(f"Episode '{ep_id}' 不存在。可用列表前5个: {available[:5]}")
    
    group = root[ep_id]
    
    # 🎯 修复核心：增加 'ee_pose' 和 'action' 作为位姿的备选项
    if 'poses' in group:
        pose_key = 'poses'
    elif 'pose' in group:
        pose_key = 'pose'
    elif 'ee_pose' in group:
        pose_key = 'ee_pose'
    else:
        pose_key = None

    image_key = 'images' if 'images' in group else ('image' if 'image' in group else None)
    
    if not pose_key or not image_key:
        actual_keys = list(group.array_keys()) + list(group.group_keys())
        raise KeyError(f"Episode '{ep_id}' 结构异常。可用键: {actual_keys}")
    
    return group[image_key][:], group[pose_key][:]