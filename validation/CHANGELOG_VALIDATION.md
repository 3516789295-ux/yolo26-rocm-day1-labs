# CHANGELOG — Windows 原生验证轮次(2026-11-02)

原始状态基线:见 `PHASE0_BASELINE.md`(全部原始文件 SHA256 与 PACKAGE_MANIFEST.json 一致)。
本文件记录本轮验证中对仓库的全部改动及理由。非 Git 仓库,差异以 SHA256 快照 + 本清单履行可追溯性。

## 0. 关键缺陷修复:MIOpen hipRTC 无法在 Windows 编译训练 kernel(版本零改动)

**现象**:GPU 冒烟第 1 轮 `miopenStatusUnknownError`——MIOpen 的
`MIOpenBatchNormFwdTrainSpatial.cpp` 等 kernel 源在 hipRTC 编译时
`miopen_type_traits.hpp:151: fatal error: 'type_traits' file not found`。
Conv/BatchNorm 训练全链路被阻断 = Windows 无法训练 Ultralytics YOLO。

**根因**(已核实):rocm-sdk-libraries 7.14.1 内的 MIOpen 在 HIP≥7.0 时无条件
`#include <type_traits>`(宿主 STL),而 hipRTC 设备编译在隔离选项下无宿主
C++ 标准库。即 [ROCm/MIOpen#3956](https://github.com/ROCm/MIOpen/issues/3956),
上游修复 = [ROCm/rocm-libraries#13437](https://github.com/ROCm/rocm-libraries/pull/13437)
(2026-10-19 合入,任务指定参考链接)。**AMD 源上无包含该修复的 7.14.x wheel**
(索引核实最高 7.14.1;TheRock bump 2026-10-20 晚于 wheel 分支)。

**本地方案(等价复刻上游修复思路,不改任何锁定版本/不改 wheel 内容)**:
1. `scripts/miopen_shims/`:4 个完全自包含的 freestanding 头
   `<type_traits>` `<utility>` `<initializer_list>` `<cstdint>`,
   以 `#if defined(__HIP__)||defined(__HIP_PLATFORM_AMD__)||defined(__HIPRTC__)`
   守护——仅在 HIP 设备编译单元生效,对宿主工具链零影响;
2. hipRTC 官方扩展点 `HIPRTC_COMPILE_OPTIONS_APPEND=-I<项目>/scripts/miopen_shims`
   (从 hiprtc0714.dll 提取的官方支持环境变量)注入上述目录;
3. `scripts/install_miopen_shims.py`:部署器(复制 shim 至 venv sdk include +
   `setx` 用户级持久化,含撤销说明)。

**验证**:同一 BatchNorm2d 复现器,修复前 FAIL(`type_traits not found`)→
修复后 `forward OK 0.5s / backward OK`;GPU 准入门禁 17/17 PASS;
6/6 Notebook(含两轮真实训练)全过。逐步排查日志:validation/logs/bn_repro_*.log。

**边界**:仅 Windows 需要此修复(Linux comgr 注入 STL);上游发布含 #13437 的
正式 wheel 后,删除 shim 目录与环境变量即可,无残留依赖。

## 1. 代码修改(1 处,最小化)

### `course/lab_utils.py` — `root()` 的 `/workspace` 分支加 POSIX 门控

```diff
-    if Path('/workspace').exists():
+    # /workspace is a Linux Radeon Cloud convention; on Windows Path('/workspace')
+    # resolves drive-relative (<drive>:\workspace) and must not hijack the default,
+    # so the project-local directory is used instead.
+    if os.name == 'posix' and Path('/workspace').exists():
         return Path('/workspace/yolo_course').resolve()
```

**理由**:Windows 上 `Path('/workspace')` 是"当前盘相对路径"(`C:\workspace`)。若机器恰好
存在该目录(或用户创建过),课程输出会被劫持到错误位置。加 `os.name == 'posix'` 后:
Linux Radeon Cloud 行为逐字不变;Windows 确定性落到 `Path.cwd()/yolo_course`(任务要求
"Windows 默认输出到项目内可控的运行目录")。`YOLO_COURSE_HOME` 优先级不变(两种 OS)。

**影响面**:仅路径选择;不改任何数据/QA/评测逻辑。回归:`tests/test_windows_compat.py::test_root_*`。

## 2. 未修改的项(及原因)

- **六份 Notebook(.ipynb)零改动**:逐 cell 审计后确认 Windows 兼容——SETUP 候选路径
  cwd/parent 优先命中、训练全程 `workers=0`、`amp=False` 课程默认、pathlib 路径、
  `/dev/kfd` 仅探测不依赖。教学目标、练习、断言、思考题全部保留。
- **`build_notebooks.py` 未改**:因 Notebook 零改动,不存在"重建覆盖人工修复"的回归面;
  生成器与成品保持一致(重建产物 SHA 应等于原始——见 NOTEBOOK_EXECUTION_MATRIX 附注)。
- **`requirements-course.txt` 未改**:以 constraints 方式保护三件套,而非改需求文件。

## 3. 新增文件

| 文件 | 用途 |
|---|---|
| `pyproject.toml` | 项目元数据 + `rocm-gfx1100` extra(表达 AMD 源与版本约束)+ uv 索引声明 |
| `constraints-rocm.txt` | 三件套精确锁定,防解析器替换 |
| `scripts/gpu_smoke.py` | 物理 GPU 准入门禁(版本/HIP/Fwd/Bwd/Optimizer/NaN/BatchNorm 探针) |
| `scripts/bn_repro.py` | MIOpen #13437 缺陷最小复现器(修复前后对照证据) |
| `scripts/miopen_shims/` + `scripts/install_miopen_shims.py` | MIOpen hipRTC freestanding STL 修复(见 §0) |
| `scripts/run_all_labs.py` | nbclient 顺序执行 00→05,逐 cell 计时、产物校验、执行数据 JSON |
| `scripts/validate_windows.ps1` | 一键验证入口(pip check → GPU 冒烟 → pytest → 全 Notebook) |
| `tests/test_windows_compat.py` | +10 项回归:路径/HIP 伪装防线/A/B 纯度/Notebook 结构/Windows 分隔符 |
| `docs/ROCM_WINDOWS_SETUP.md` | Windows 从零搭建指南 |
| `docs/RADEON_CLOUD_LINUX_SETUP.md` | Linux Radeon Cloud 部署指南(含与 Windows 验证关系) |
| `docs/TROUBLESHOOTING.md` | 分类故障排查(含首次 MIOpen 编译慢、#13437 参考) |
| `validation/*` | 本轮全部证据(基线、日志、执行矩阵、最终报告) |

## 4. 数据预置(非代码改动)

- `yolo_course/data/raw/coco128.zip`:按 README 官方离线预案预置
  (SHA256 `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`,
  下载自官方 `ultralytics/assets` v0.0.0 release,与本轮直连下载一致)。
  Lab B 检测到压缩包后解压复用,不触发再次下载。
- `yolo26n.pt` 不预置:Notebook 00 走真实下载路径(本轮网络已验证可达)。

## 5. 回归测试结果

- 原始测试 6 项 + 新增 10 项 = **16 passed**(Windows 原生,`.venv`,pytest 9.2.0)。
- `pip check`:无冲突;三件套版本未漂移。

## 6. 双语版再生成(2026-11-02 第二轮,公开发布版)

- `build_notebooks.py` 重写为双语生成器:`m(zh, en)` 输出中文 + `---` + 英文;
  标题/横幅双语;**全部代码 cell 与已验证 v1.0 字节级一致**(hash 比对见
  `validation/logs/code_cells_baseline.json`,6/6 True)。
- 双语版全量 GPU 复验(隔离目录 `--course-home yolo_course_bilingual_check`):
  **6/6 PASS,37/37 cells**;A/B 指标与首轮逐位一致(P/R/mAP50/mAP50-95 完全相同),
  数据 `NOTEBOOK_EXECUTION_DATA_20261102_210903.json`。
- 新增仓库发布文件:README.md(重写为仓库主页)+ README.en.md、CONTRIBUTING.md(双语)、
  LICENSE(MIT + 第三方声明)、.gitignore、.github PR/Issue 模板、
  docs/*.en.md ×3、INSTRUCTOR_GUIDE.en.md、validation/FINAL_REPORT.en-summary.md。
- SHA256SUMS 按发布树重生成(原 v1.0 快照保留于 PACKAGE_MANIFEST.json 与 §7 基线)。
