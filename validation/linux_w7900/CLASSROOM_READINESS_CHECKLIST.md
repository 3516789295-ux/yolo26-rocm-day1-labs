# CLASSROOM_READINESS_CHECKLIST.md — 28 人 / 5 组 · W7900 课堂发布清单

- 适用：Linux · AMD Radeon Pro W7900（gfx1100, 48GB）· uv 隔离栈（torch 2.12.0+rocm7.14.1）
- 用法：按时间倒序执行；每项有负责人与验证方法。勾选框供打印或复制到讲师备忘。
- 姊妹文档：[03_CLASSROOM_CAPACITY_PLAN.md](03_CLASSROOM_CAPACITY_PLAN.md)（容量测算）、[INSTRUCTOR_GUIDE.md](../../../INSTRUCTOR_GUIDE.md)（教学秩序）

## A · 发布前一周（T-7）

- [ ] **权限与素材**：讲师准备可合法使用的标注图片（每人 ≥5 张；CC0/自拍/授权场景），发布类别规范（classes.txt 三类顺序统一）。
- [ ] **学生课前作业**（微信群/学习平台下发）：①自带笔记本安装 X-AnyLabeling（不装入 ROCm 环境）；②观看一次"导出 YOLO Detection txt"演示录屏；③按规范标注 ≥5 张并同伴复核 5 张。
- [ ] **验收方法**：抽 3 份学生导出（images/ + labels/ + classes.txt 齐全、文件名一一对应）视为作业合格样本。

## B · 发布前一天（T-1）

- [ ] **环境就绪**：课堂机按 `docs/RADEON_CLOUD_LINUX_SETUP.md` §2B 建 uv 隔离栈（或使用已验证镜像）。
- [ ] **运行 Preflight**（本仓库新增工具，只读不装）：
  ```bash
  .venv/bin/python scripts/preflight_classroom.py --course-home /workspace/classroom_demo --json preflight_T-1.json
  ```
  **11/11 PASS 才继续**；任何 FAIL 按输出的 next 提示处理。JSON 留档。
- [ ] **资产预置**（若课堂网络不通 github release）：
  | 资产 | 位置 | 官方 SHA256（前 16 位） |
  |---|---|---|
  | yolo26n.pt | 项目根目录 | `9b09cc8bf347f0fc` |
  | coco128.zip | `<course_home>/data/raw/coco128.zip` | `61e5e3028863d8ff` |
  | coco8（16 文件） | `<datasets_dir>/coco8/{images,labels}/{train,val}/` | —（官方 Ultralytics/COCO8） |
  受限网络取数路径与对账方法：[00_BASELINE_AUDIT.md](00_BASELINE_AUDIT.md) §4。
- [ ] **Kernel 预注册**：`.venv/bin/python -m ipykernel install --user --name yolo26-rocm7141 --display-name "YOLO26 - AMD ROCm 7.14.1"`（Preflight 第 7 项会复核 argv 指向）。
- [ ] **预热一次**：用与学生相同的 Kernel 跑一遍 `run_all_labs.py --course-home /workspace/classroom_demo --practice-fill <讲师填样>`（~3–5 分钟）。作用：①消除 MIOpen JIT 冷启动（否则首个学生训练多等 ~1 分钟）；②生成**讲师预制工件**（Plan B 降级用，见 E 节）；③验证全链路。
- [ ] **目录规划**（勿用 `group{1..5}` 花括号——那是字面字符串）：
  ```bash
  # 每组各设各的，例如第 2 组在自己终端执行：
  export YOLO_COURSE_HOME=/workspace/group2_course
  # 学生个人再隔离：/workspace/group2_student07
  ```

## C · 当天课表（GPU 错峰，总课时 ≈ 4 小时）

| 时段 | 环节 | GPU 占用 | 并发上限 | 备注 |
|---|---|---|---|---|
| 0:00–0:15 | 00 环境预检 | 每人 ~5 s 突发 | 全员 | Preflight 已保证 |
| 0:15–0:35 | Lab A 推理阈值 | 每人 ~10 s | 全员 | 推理突发可并行 |
| 0:35–1:30 | Lab B 标注与 QA | 0（CPU） | 全员 | 标注在学生笔记本上做 |
| 1:30–2:30 | Lab C 训练 | 40–100 s/人 | **≤3** | 排队看曲线也计入学习 |
| 2:30–3:10 | Lab D 评价分析 | ~15 s/人 | 全员 | 分析为主 |
| 3:10–3:55 | Lab E A/B | 30–65 s/组 | **1（独占）** | **计时轮禁止任何并发** |
| 3:55–4:00 | 收尾：整理 GitHub 提交清单 | 0 | 全员 | 用学生模板 |

> A/B 独占的执行方式：讲师口头点名轮转，"第 3 组准备→开始→完成"；其余组此时做 Lab D 归因或写假设。

## D · 常见课堂阻塞与 30 秒处置

| 症状 | 30 秒处置 |
|---|---|
| 学生说"全红跑不动" | 助教看 00 的报错：`hip=None`→换 Kernel；`NMS unavailable`→用隔离栈环境 |
| 训练"卡住不动" | 指等待说明（首次 ~1 分钟 JIT）；`rocm-smi` 确认 GPU 在跑 |
| X-AnyLabeling 导入报错 | 报错里有文件名与行号；对照 Lab B 错误表逐项修 |
| Lab E 计时被污染 | 该组重跑该臂，报告注明干扰 |

## E · 降级路径（Plan B，明确来源、不冒充）

触发条件：环境突发故障 / 网络中断 / 课时压缩 >30 min。

1. 学生仍亲手完成：00、Lab A、Lab B 全链、Lab C 的 COCO8 冒烟、Lab D 归因、Lab E 假设撰写。
2. 长训练与 A/B 使用**讲师预制工件**：T-1 预热运行的 `/workspace/classroom_demo/runs/**` 与 `reports/**`。
3. **标注规范**：学生引用时必须写明"讲师预制工件（2026-11-XX 预热运行，SHA256 见 reports/）"；评分表训练分按冒烟计。
4. 禁止表述："全班完成了真机完整训练"。

## F · 3 名真实学生试走验收表（Student Learning 去 PROVISIONAL 的前置）

> 在发布前或首课后，选 3 名**未参与开发**的学生（建议 1 名基础弱 / 1 名中等 / 1 名较强）独立试走，讲师只观察不代操作。

| 记录项 | 学生 1 | 学生 2 | 学生 3 |
|---|---|---|---|
| 姓名/自评 Python 水平 |  |  |  |
| 独立完成率（6 份中完成几份） | __ / 6 | __ / 6 | __ / 6 |
| 求助次数（含问什么） |  |  |  |
| 00→01 用时 |  |  |  |
| 02（含标注导入）用时 |  |  |  |
| 03→05 用时 |  |  |  |
| 练习格是否真实填写（TODO 断言触发过几次） |  |  |  |
| 卡点 1（现象/页面/报错原文） |  |  |  |
| 卡点 2 |  |  |  |
| 学生一句话建议 |  |  |  |

**通过标准**：3 人平均独立完成 ≥5/6；人均求助 ≤3 次；无"看不懂下一步"类卡点。未达标项回填 P1 修复后再发布正式版。

## G · 发布判定

- [ ] A–C 全部勾选 + Preflight 11/11 → **GO（Plan A）**
- [ ] A–C 有未勾选但 E 路径就绪 → **CONDITIONAL GO（Plan B）**
- [ ] Preflight 有 FAIL 且不可当日修复 → **NO-GO，改讲座 + 演示**
- [ ] F 未执行 → Student Learning 结论保持 **PROVISIONAL**（不得宣称"经学生验证"）
