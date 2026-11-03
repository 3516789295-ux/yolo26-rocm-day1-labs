# 07 · Defects & Fixes — Linux W7900 Validation Round

- 记录时间（UTC）：2026-11-03 03:15
- 分支：`linux-w7900-validation-fixes`（本地提交，未经 Owner 批准不推送）
- 修复原则：最小有效修改、经生成器单一事实源、每项配自动化回归、旧证据不删改

## P0（阻碍执行 / 破坏结论 / 违反门禁）

### P0-A · AST Reference 损坏（Lab B）
- **现象**：main 上 `pytest` 1 失败——`test_notebook_code_matches_ast_reference`：Lab B cell 1 摘要不匹配、cell 5 "摘要" 仅 54 字符。
- **根因**：X-AnyLabeling 功能进入 cell 1 后参考未再生（陈旧摘要 `cdac730a…`）；cell 5 被手工补丁成截断摘要（`760476bc…` 54 字符，无有效 SHA-256 可能匹配）。另发现提交版 Lab B 与生成器差一行注释（字节不同步）。
- **修复**：①独立临时目录重建六份 Notebook 并与提交版逐格对比（5/6 完全一致，Lab B 仅差 AST 等价的注释行）→ 以生成器为准再生全部 Notebook；②用测试模块自带的 `--regenerate` 路径再生参考（统一为 raw canonical AST 字符串，6 份/41 格）；③新增 `test_reference_digest_format`：dict 型摘要必须为 64 位小写十六进制（截断/手补丁直接红）。
- **证据**：修复前 `1 failed, 19 passed, 1 skipped` → 修复后（含全部新测试）`39 passed, 1 skipped`。
- **提交**：`8fc7cb8`（初版 37 格）、`81330da`/后续（41 格）。

### P0-B · run_all_labs.py Lab B 工件检查硬编码 COCO128 路径
- **现象**：`REQUIRED_ARTIFACTS['02']` 固定检查 `data/processed/coco128_classroom_3class/data.yaml`——X-AnyLabeling / 用户 YAML 路径假失败；复用旧 course home 时旧残留假通过。
- **修复**：新增 `_verify_lab_b_dataset()`：从 `course_config.json` 读取本轮实际 `dataset_yaml + dataset_kind`；三种模式（COCO128 / X-AnyLabeling / user_provided）分别校验位置与专属证据（X-AnyLabeling 需 import 报告）；**新鲜度门禁**：config 与 QA 报告必须在本 runner 会话内重写（mtime ≥ 会话起点），杜绝旧残留假 PASS；QA `pass=false` 判 FAIL。
- **回归**：`tests/test_run_all_labs_gates.py` 9 项（三模式接受/拒绝、陈旧拒绝、QA 失败、跨 home 数据集拒绝、探针代码执行）；真实运行双重验证（主链 COCO128 模式 + B2 X-AnyLabeling 模式均按配置校验通过）。

### P0-C · Kernel 回退不验证解释器归属
- **现象**：找不到注册 Kernel 时回退 `python3`，仅凭名称存在即运行——本机平台栈恰好是现实反例（torchvision 缺 GPU NMS，Lab A 必死）。
- **修复**：新增 `probe_kernel()`：正式运行前**真实启动**将使用的 Kernel，执行探针返回 `sys.executable / torch.__version__ / torch.version.hip / torch.cuda.get_device_name(0)`；四项全部属于已验证 ROCm 环境才放行，否则 exit 2 并给出注册命令；探测结果写入执行汇总 JSON。
- **证据**：三轮真实运行 `kernel_probe.ok=True`（executable=`.venv/bin/python`，gpu=W7900D）；`tests/test_run_all_labs_gates.py::test_kernel_probe_code_executes_and_emits_marker`。

### P0-D · runner `--only` 部分运行退出码错误（本轮新发现）
- **现象**：成功完成 `--only 02,03` 仍 exit 1（分母用了全部 6 份）。
- **修复**：与 `len(selected)` 比较；`--only 00` 实测 exit 0。
- **证据**：B2 运行日志（修复前 2/2 PASS 但 exit 1）→ 修复后实测。

## P1（学生无法按文档独立完成 / 关键概念错误）

### P1-A · 五份 Notebook 英文时长/学习路径交换
- **现象**：00/01/03/04/05 的中文"学习路径"显示英文时长（如"学习路径：10–15 minutes"），英文 "Suggested time" 显示中文路径链。
- **根因**：`make()` 调用点 `duration_en` 与 `tags_zh` 实参交换（仅 02 正确）。
- **修复**：五个调用点参数归位；`make()` 内加防错断言（时长必须含时间单位、路径不得含）；再生成 Notebook；新增 `test_notebook_metadata.py`（双语头部字段解析 + 交换检测）。
- **证据**：再生 diff 仅 5 个头部 Markdown 单元变化（见提交 `8fc7cb8`）。

### P1-B · 教学要素与学生操作环节缺失（Phase 5 审计产出）
- **现象**：00/03/04 无学生主动操作环节；多份缺 Prerequisites/Expected Output/Common Errors/Interview Questions/Takeaway；Lab A IoU 练习为"改了就看"无预测门槛；Lab E 无假设先行环节。
- **修复**（全部经 `build_notebooks.py` + AST 再生 + 测试）：
  - 00：预测-运行-对照练习（MY_PREDICTION + TODO 门禁，存 `lab00_practice.json`）
  - 01：IoU 练习升级预测先行（MY_PREDICTED_IOU + 门禁）
  - 03：读自己的训练曲线（MY_READING，须引用自己 results.csv 数字）
  - 04：单错误四字段归因（现象/证据/假设/验证，MY_ANALYSIS）
  - 05：假设在 Variant 训练前封存（MY_HYPOTHESIS）+ 结果后"对照你的假设"节
  - 元数据补全：Prerequisites/预期输出/常见错误表/面试题/Takeaway（详见 06_SCORECARD §4）
  - 无人值守复验通道：`YOLO_PRACTICE_FILL`（runner `--practice-fill`，注入值在输出原文可见并记录于汇总）——学生门禁不放松，TODO 断言默认必触发
- **回归**：`test_every_notebook_keeps_a_student_action_cell`（严格 TODO 门禁；Lab B 标注修正为安全 opt-in 豁免但强制审计）+ `test_generated_code_cells_compile`。
- **实证**：无人值守运行缺填充时 Lab A 真实 FAIL（`20261103_024247`）——门禁不是摆设。

## P2（体验改进）

- P2-1 学生项目 README 模板缺失 → 新增 `templates/STUDENT_PROJECT_README_template.md`（占位符防虚构指标 + 面试陈述结构）。
- P2-2 02/04/05 的 Common Errors 未成表（内联异常已兜底）→ 遗留建议。
- P2-3 训练等待期无状态说明、Lab E 图表单指标、Kernel 切换指引缺截图 → 遗留建议（见 06_SCORECARD §6）。

## 本轮验收过程中自行引入并修复的缺陷（如实记录）

| 缺陷 | 引入点 | 发现方式 | 修复 |
|---|---|---|---|
| 生成器两行行首多一个空格 → 05 两个 cell IndentationError | P1-B 编辑时 | AST 再生 `ast.parse` 报错 | 删空格 + 新增 `test_generated_code_cells_compile` 防同类 |
| B2 fixture scenes.csv 用原始文件名 | fixture 构建 | **导入器场景覆盖守卫**在真实 Notebook 路径拦截（`20261103_022724` FAIL） | 修正 fixture（该 FAIL 保留为异常注入有效性证据） |
| 练习填充 JSON 漏 lab_a_iou 键 | 最终运行准备 | 最终轮首试 Lab A FAIL（TODO 门禁触发） | 补键 + 01 练习格补注入钩子；该 FAIL 保留为门禁实证 |

## 环境级发现（非仓库缺陷，已记录处置）

| 发现 | 处置 |
|---|---|
| 平台预装 torchvision（0.24.1+d801a34）缺 GPU NMS 内核 | 与 `docs/RADEON_CLOUD_LINUX_SETUP.md` §2A 记录一致；采用方式 B uv 隔离栈；平台环境未改动 |
| 出口代理阻断 github release 资产 / ultralytics.com 图片 | 资产经 hf-mirror 预置并逐一 SHA256 对账官方值（00_BASELINE_AUDIT §4）；课堂预案写入 03_CAPACITY_PLAN §6 |
| best.pt 文件哈希跨运行不同（内嵌时间戳元数据），但指标同种子 4 位小数一致 | 02_AUDIT §9 如实记录口径：以指标与 state_dict 数值论复现性，不以文件字节论 |

## 测试汇总

| 套件 | main 基线 | 修复后 |
|---|---|---|
| pytest | 19 通过 / **1 失败** / 1 跳过 | **39 通过 / 0 失败 / 1 跳过**（新增 20 项回归） |
| pip check | — | 通过（rc=0） |
| gpu_smoke.py | — | 17/17 门禁 PASS |
| run_all_labs.py 最终轮 | — | 6/6 PASS · 41/41 cells · exit 0 |
