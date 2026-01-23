# Change: 训练时自动同步数据并移除数据中心菜单

## Why
当前训练流程在无数据或数据不足时需要手动同步，容易失败；同时数据中心入口不再需要保留。

## What Changes
- 训练触发时若数据为空或低于阈值，自动同步数据后再训练
- 在导航菜单中移除“数据中心”入口

## Impact
- Affected specs: qlib-ml-strategy, mvp-features
- Affected code: 训练任务/数据同步流程、前端导航模板
