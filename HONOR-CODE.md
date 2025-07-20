# Honor Code

## 代码参考声明

本项目在开发过程中参考了 [haimengzhao/juno-simulation](https://github.com/haimengzhao/juno-simulation/blob/master) 项目的部分实现方法和优化策略。具体参考内容如下：

### 1. 空间索引优化算法
**参考文件**: `genPETruth.py`  
**参考内容**: KDTree空间索引的实现思路和PMT快速查找策略  
**应用位置**: `simulate.py` 中的 `OptimizedPMTSimulator.__init__()` 和 `kdtree_prefilter()` 方法  
**参考原因**: 学习如何使用KDTree数据结构优化大规模PMT阵列的空间查找效率

### 2. 几何预筛选策略
**参考文件**: `genPETruth.py` 和 `README.md`  
**参考内容**: 基于角度约束的PMT预筛选算法思路  
**应用位置**: `simulate.py` 中的 `geometric_prefilter()` 方法  
**参考原因**: 提高光子在PMT表面相互作用的物理准确性

### 3. 分层批处理优化
**参考文件**: `README.md`  
**参考内容**: 文档中描述的分块处理策略和内存优化方案  
**应用位置**: `simulate.py` 中的 `process_pmt_batch()` 方法和批处理逻辑  
**参考原因**: 学习如何处理大规模数据时的内存管理和性能优化  

## 致谢

感谢 [haimengzhao/juno-simulation](https://github.com/haimengzhao/juno-simulation) 项目提供的优秀参考实现，为本项目的优化提供了宝贵的技术指导。
