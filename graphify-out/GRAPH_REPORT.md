# Graph Report - /home/xk/coding/M1/Go2Pvcnn/go2_pvcnn/assets  (2026-08-14)

## Corpus Check
- Corpus is ~1,876 words - fits in a single context window. You may not need a graph.

## Summary
- 34 nodes · 37 edges · 7 communities (4 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Robot Asset Configuration
- Robot Asset Configuration
- Robot Asset Configuration
- Robot Asset Configuration
- Robot Asset Configuration
- Robot Asset Configuration
- Robot Asset Configuration

## God Nodes (most connected - your core abstractions)
1. `GlobalRigidObjectCollection` - 10 edges
2. `GlobalRigidObject` - 6 edges
3. `GlobalRigidObjectCollectionCfg` - 4 edges
4. `Configuration for Unitree Go2 robot for PVCNN project.  This module provides rob` - 1 edges
5. `支持多环境训练的全局刚体对象          A rigid object that exists as a single instance shared a` - 1 edges
6. `初始化全局刚体对象                  Initialize the global rigid object.          Args:` - 1 edges
7. `重置外部力和力矩（针对全局对象优化）                  Reset external wrench (optimized for global` - 1 edges
8. `返回实例的信息字符串 Returns: A string containing information about the instance.` - 1 edges
9. `全局刚体对象集合          Global rigid object collection for multi-environment training.` - 1 edges
10. `初始化全局刚体对象集合                  Initialize the global rigid object collection.` - 1 edges

## Surprising Connections (you probably didn't know these)
- `GlobalRigidObjectCollectionCfg` --uses--> `GlobalRigidObjectCollection`  [INFERRED]
  rigid_object_collection/global_rigid_object_collection_cfg.py → rigid_object_collection/global_rigid_object_collection.py

## Import Cycles
- None detected.

## Communities (7 total, 3 thin omitted)

### Community 0 - "Robot Asset Configuration"
Cohesion: 0.22
Nodes (6): 重置物体集合状态                  Reset the rigid object collection state., 写入物体位姿到仿真                  Write object link pose to simulation., 写入物体速度到仿真                  Write object link velocity to simulation., 重置外部力和力矩（针对全局对象优化）                  Reset external wrench (optimized for global, slice, Tensor

### Community 1 - "Robot Asset Configuration"
Cohesion: 0.22
Nodes (6): GlobalRigidObject, 返回实例的信息字符串 Returns: A string containing information about the instance., 支持多环境训练的全局刚体对象          A rigid object that exists as a single instance shared a, 初始化全局刚体对象                  Initialize the global rigid object.          Args:, RigidObject, RigidObjectCfg

### Community 2 - "Robot Asset Configuration"
Cohesion: 0.50
Nodes (3): GlobalRigidObjectCollection, 全局刚体对象集合          Global rigid object collection for multi-environment training., RigidObjectCollection

### Community 3 - "Robot Asset Configuration"
Cohesion: 0.67
Nodes (3): GlobalRigidObjectCollectionCfg, RigidObjectCollectionCfg, 全局刚体对象集合的配置          Configuration for global rigid object collections.

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GlobalRigidObjectCollection` connect `Robot Asset Configuration` to `Robot Asset Configuration`, `Robot Asset Configuration`, `Robot Asset Configuration`, `Robot Asset Configuration`?**
  _High betweenness centrality (0.566) - this node is a cross-community bridge._
- **Why does `GlobalRigidObject` connect `Robot Asset Configuration` to `Robot Asset Configuration`?**
  _High betweenness centrality (0.394) - this node is a cross-community bridge._
- **Why does `GlobalRigidObjectCollectionCfg` connect `Robot Asset Configuration` to `Robot Asset Configuration`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._