# 03 · Classroom Capacity Plan — 28 Students / 5 Groups on Linux W7900

- 基准数据：本轮真实测量（W7900D 48GB · ROCm 7.14.1 栈 · 128 图教学集 · 12 Epoch · imgsz512/416 · batch8）
- 结论速览：**一块 W7900 足以支撑 Plan A（全员真机）**——实测 12-Epoch 训练仅 25–65 s，全链 6 份 Notebook 热启动 ~95 s/组。主要课堂风险不在算力，在**网络受限的资产预置**与 **Lab B 桌面标注的课前准备**。

## 1. 实测时间与资源基线（本轮真机数据）

| 项目 | 冷启动（当日首次） | 热启动（MIOpen 缓存后） | 峰值显存 |
|---|---|---|---|
| 全链 00→05（含两次 12-Epoch 训练） | 195 s | **93 s** | ≤1.54 GiB |
| Lab C Notebook（COCO8 冒烟+12Epoch） | 100 s | 40 s | 1.54 GiB |
| Lab E Notebook（Variant 12Epoch+双评测） | 67 s | 31 s | 1.07 GiB |
| 单次 12-Epoch baseline 训练 | 63.6 s | **29.3 s** | 1.54 GiB |
| 单次 12-Epoch variant 训练 | 61.2 s | 25.3 s | 1.07 GiB |
| Lab A 推理（3 阈值） | 7.3 s | ~5 s | <1 GiB |
| Lab B 数据 QA（CPU） | 2.6 s | 2.6 s | — |
| Lab D 官方验证+逐图匹配 | 13.4 s | 7.9 s | <1 GiB |
| MIOpen JIT 预热 | ~60 s（一次性，全机共享磁盘缓存） | 0 | — |
| 资产下载（受限网络下需预置） | yolo26n.pt 5.5MB + coco128.zip 7MB + coco8 1MB | — | — |

**关键事实**：
1. 显存峰值 1.54 GiB vs 48 GB——**显存完全不构成瓶颈**（理论可容 20+ 并发训练）。
2. 瓶颈是**计算单元争用**：并发训练会互相拖慢且污染 A/B 计时（Lab E 要求同卡同条件）。
3. 同种子完全可复现（指标 4 位小数一致）——小组间结果可比、可复核。
4. 训练时长 25–65 s 意味着"排队"成本极低：28 人逐个跑 Lab C 也只需 ~20 分钟 GPU 时间。

## 2. 一块 W7900 的调度方案

### 2.1 目录与资产隔离（防互相覆盖）

```bash
# 每组（或每人）设置一次自己的目录——注意：不要写 group{1..5} 花括号展开，
# 那会把字面字符串 "{1..5}" 赋给变量。正确写法是每组各设各的，例如第 2 组：
export YOLO_COURSE_HOME=/workspace/group2_course        # 组号由讲师分配：group1..group5
# 或 runner 显式传参（等价）：
#   python scripts/run_all_labs.py --course-home /workspace/group2_course
# 学生个人再隔离：/workspace/group2_student07
# 模型权重共享只读缓存（00 会各自复制到本组 artifacts/，SHA256 一致性由断言保证）
```
- 组间零冲突：config、runs、reports 全部按 home 隔离；数据集各自重建（2.6 s，代价可忽略）。
- 学生个人再隔离：`YOLO_COURSE_HOME=/workspace/g1_student07`。

### 2.2 错峰排程（实测推算）

| 时段 | 活动 | GPU 占用 | 说明 |
|---|---|---|---|
| 课前 | 讲师预热一次全链 + 预置资产 | ~4 min | 消除 MIOpen JIT 冷启动；验证网络 |
| 上午 | 00 + Lab A | 每人 <10 s 突发 | 28 人串行推理共 ~7 min；讲解为主 |
| 上午 | Lab B（标注在学生笔记本上完成，见 §4） | 0（CPU QA） | 全员同时跑 QA 无压力 |
| 下午 | Lab C | 每组/每人 40–100 s | **建议串行**：5 组 ≈ 8 min；28 人逐个 ≈ 20 min |
| 下午 | Lab D | 每人 <15 s | 评测突发即可；分析是人力活 |
| 下午 | Lab E | 每组 31–67 s | **A/B 计时期间必须独占 GPU**（公平性）；5 组 ≈ 5 min |

### 2.3 并发策略

- **允许并发 2–3 个训练**（显存充裕），但 **Lab E 计时轮必须独占**——并发会把对方当"云端负载干扰"，破坏"相同 ROCm GPU"条件。
- 无 GPU 可用时学生的学习路径：Lab B 标注/QA、Lab D 错误归因（读已有 CSV/图）、Lab E 假设撰写——全部可离线推进，GPU 仅在最后验证时需要。

### 2.4 冷/热启动差异（必须向学生说明）

首次训练含 ~60 s kernel 编译（全机一次性）；学生若在课上第一次触发会误以为"卡死"。**讲师课前预热 + 首页说明**可完全避免。实测对照已写入 02_MODEL_AND_DATA_AUDIT.md §9。

## 3. Plan A — Full Hands-on（推荐，实测可行）

**每人自己的权重、自己的曲线、自己的 A/B 结论。**

| Lab | 课堂时长（含讲解） | GPU 时间/人 | 28 人规模推算 |
|---|---|---|---|
| 00 环境预检 | 15 min | ~5 s | 并行无压力 |
| A 推理与阈值 | 20 min | ~10 s | 串行 7 min |
| B 数据与 QA | 45–60 min | 0 | CPU 并行 |
| C 训练微调 | 60 min | 40–100 s | 28 人串行 ≤20 min（或 5 组代表跑 + 其他人看实时曲线） |
| D 评价分析 | 40 min | ~15 s | 评测突发 |
| E A/B 实验 | 40 min | 31–67 s/组 | 5 组独占轮转 ≈5 min |

- 合计课时 ≈ 4 h（与 README/讲师指南的节奏表一致），GPU 总需求 <45 min。
- 前提：①资产已预置（§4）；②Kernel 已预注册；③讲师已预热。

## 4. Plan B — Classroom-Safe（备用方案，不冒充 Plan A）

**适用**：网络/环境突发故障、学生数据集变大（如 500+ 图 × 更长训练）、课时被压缩。

- 学生仍亲手完成：00、Lab A、Lab B 完整链（标注+QA+防泄漏）、Lab C 的 COCO8 冒烟（~15 s）、Lab D 用**讲师课前封存的 best.pt** 做全部分析、Lab E 假设撰写 + 对照讲师封存的 A/B 工件。
- 讲师课前封存：跑一次完整链（~4 min），产物标记"讲师预制工件"（INSTRUCTOR_GUIDE 已有此要求），学生引用时**必须注明来源**。
- **Plan B 不产生 Plan A 的"我自己训的 best.pt"证据**——评分标准相应调整（训练与复现 20 分中冒烟占 10，分析占 10）。
- 明确标注：Plan B 的成绩单不得声称"全员真机完整训练"。

## 5. Lab B 桌面标注的课前准备清单（最大阻塞源）

| 项 | 要求 | 建议时限 |
|---|---|---|
| X-AnyLabeling 安装 | 学生自带笔记本（Win/mac/Linux），课前作业装好；**不装入** ROCm 环境 | 课前 2 天 |
| 授权图片素材 | 讲师准备每人 ≥5 张可合法使用的图（CC0/自拍场景等）；禁止随意网图 | 课前 1 周 |
| 类别规范 | classes.txt 三类顺序统一（建议 person/car/chair 或业务类） | 课前发布 |
| 导出演练 | Export Annotations → YOLO Detection → txt；一次课堂演示录屏 | 课前 1 天 |
| 上传通道 | Radeon Cloud 上传 images/ labels/ classes.txt（+可选 scenes.csv） | 课上 10 min |
| 同伴复核 | 5 张交叉复核（INSTRUCTOR_GUIDE 评分项） | 课上 15 min |

## 6. 网络受限环境的资产预置（本轮实测路径）

本平台 github release 资产被出口代理阻断，实测可行的预置链（全部 SHA256 校验）：

1. `yolo26n.pt` ← hf-mirror `openvision/yolo26-n`（SHA256 = 官方 `9b09cc8b…`）→ 放项目根目录
2. `coco128.zip` ← hf-mirror `LibreYOLO/coco128`（SHA256 = 官方 `61e5e302…`）→ 放 `<course_home>/data/raw/`
3. `coco8` ← HF 官方 `Ultralytics/COCO8` 16 文件 → `/workspace/datasets/coco8/`
4. 讲师以 `run_all_labs.py` 预热并留档。

> 若课堂网络开放（可达 github），Notebook 自带下载路径可直接用，无需预置。

## 7. 课堂最常见阻塞与预案（按概率排序）

1. **Kernel 选错/未注册**（高）→ 讲师课前统一注册 + 00 的常见错误表自查；助教 30 秒判别（`torch.version.hip`）。
2. **平台 torchvision 缺 GPU NMS**（中，镜像相关）→ `require_rocm()` 首格即拦截并指向部署文档 §2B；讲师备好隔离栈镜像。
3. **X-AnyLabeling 导出格式错**（中）→ 导入器报错精确到文件（本轮真实触发过场景覆盖缺口，报错可直接定位）；备 COCO128 fallback 保底完成教学目标。
4. **首次训练"看起来卡住"**（中）→ 冷启动说明 + 课前预热。
5. **网络下载失败**（低，已预置）→ §6 预置链。
6. **A/B 计时被并发污染**（低）→ E 轮独占调度。

## 8. 判定

**Classroom Readiness：GO（条件：完成 §5 课前清单 + §6 资产预置 + 讲师预热）。** 实测算力余量极大（GPU 总需求 <45 min / 4 h 课时；峰值显存 1.5/48 GiB）；主要工程风险已由预置与门禁覆盖。
