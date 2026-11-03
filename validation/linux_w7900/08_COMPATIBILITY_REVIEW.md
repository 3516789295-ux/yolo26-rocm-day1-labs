# 08 · Compatibility Review — Linux W7900 Round

- 评审时间（UTC）：2026-11-03 03:20
- 范围：Linux ↔ Windows 路径隔离、文档准确性、平台矩阵、依赖约束有效性

## 1. Linux / Windows 代码路径隔离（逐文件实测）

| 文件 | 隔离机制 | 实测结论 |
|---|---|---|
| `course/lab_utils.py::root()` | `YOLO_COURSE_HOME` > POSIX `/workspace` 检测 > cwd；Windows 不会被 `C:\workspace` 劫持 | ✓ Linux 本轮以 `--course-home` 显式隔离运行；测试 `test_root_workspace_only_on_posix` 在本机 POSIX 分支通过 |
| `scripts/run_all_labs.py` | venv 路径 `Scripts/python.exe`（nt）/`bin/python`（POSIX）；MIOpen shim 环境变量仅 win32 注入 | ✓ 本轮 POSIX 分支全程无 Windows 依赖 |
| `scripts/gpu_smoke.py` | 驱动信息：Windows PowerShell WMI / POSIX `rocm-smi`+sysfs；shim 仅 win32；版本门禁 Windows 严格 / POSIX 记录放行 | ✓ POSIX 分支 17/17 PASS，`driver` 字段含 rocm-smi 输出与 kfd 节点 |
| `scripts/install_miopen_shims.py` / `miopen_shims/` | Windows 专用（hipRTC freestanding 头） | ✓ 本轮 Linux 未触发；Linux ROCm 7.14.1 wheel 自包含，BN 训练模式实测通过（无需 shim） |
| `scripts/validate_windows.ps1` | Windows 专用一键复验 | Linux 等价物 = `gpu_smoke.py` + `pytest` + `run_all_labs.py`（docs 已载明）✓ |
| Notebook SETUP cell | `candidates` 探测 cwd/父目录；POSIX 下 `/workspace` 命中 | ✓ 本轮 kernel cwd=项目根，路径解析正确 |

**判定：双平台路径完全隔离，无交叉污染。**

## 2. 文档准确性抽查（对照本轮实测）

| 文档声明 | 实测 | 判定 |
|---|---|---|
| README："Linux 侧为静态兼容核查，尚未在真实云 GPU 复验" | 本轮完成真机复验——**该声明此后过期**，建议 Owner 合并后更新为"Linux W7900 已实测" | △ 待更新（见 FINAL_REPORT 行动项） |
| `docs/RADEON_CLOUD_LINUX_SETUP.md` §2A 平台栈 NMS 缺陷实测限制（torch 2.9.1+gitff65f5b / torchvision 0.24.1+d801a34） | 与本机平台栈**逐字一致**，`NotImplementedError` 复现 | ✓ 预警准确 |
| §2B uv 自建命令 | 逐字执行成功（三件套 2.12.0+rocm7.14.1 + 课程依赖 + Kernel 注册 + pip check rc=0） | ✓ |
| §3 路径表（YOLO_COURSE_HOME / /workspace / YOLO_USER_DATA_YAML / 设备节点） | 全部实测符合 | ✓ |
| §5 workers=0 / amp=False 默认 | 本轮训练实测一致 | ✓ |
| `constraints-rocm.txt` 三件套钉版 | 安装期间无任何替换；pip check 通过 | ✓ |
| INSTRUCTOR_GUIDE 课前预演清单 | 本轮逐项可执行（权重 SHA256、预置资产、预制工件标记、错峰建议） | ✓ |
| TROUBLESHOOTING（MIOpen hipRTC / NMS / Kernel） | 与本轮实测现象一致 | ✓ |

## 3. 平台矩阵（本轮更新）

| 维度 | Windows（历史验证） | Linux W7900（本轮验证） |
|---|---|---|
| GPU | RX 7900 XTX（gfx1100, 24GB） | **Radeon Pro W7900D（gfx1100, 48GB）** |
| OS | Windows 11 原生 | Ubuntu 24.04.4 · kernel 6.8.0-79 |
| 栈 | torch 2.12.0+rocm7.14.1（repo.amd.com wheel） | **同版本，同源** |
| MIOpen | 需 freestanding shim（rocm-libraries#13437） | 无需 shim（自包含） |
| Kernel | yolo26-rocm7141（注册） | 同名注册 + **真实执行探测**（本轮新增硬门禁） |
| 执行 | 6/6 PASS（37 cells） | **6/6 PASS（41 cells，含学生练习）** |
| 指标 | mAP50-95 .380（val） | .383（同种子交叉印证） |

## 4. 依赖与许可兼容

- Python 3.12（<3.13 上限）✓；ultralytics 8.4.199 满足 `>=8.4,<9` ✓
- PyPI 依赖经 constraints 保护未触碰 AMD 三件套 ✓
- 新增文件（4 个测试 + 1 个模板 + 练习单元）不引入任何新第三方依赖 ✓
- 许可：MIT 仓库代码；权重/数据资产不入库惯例未破坏（本轮新增工件同样不含 *.pt/数据集）✓

## 5. 已知不兼容/限制（如实列出）

1. **平台预装栈不可用于检测类 Lab**（torchvision 无 GPU NMS）——文档已预警，方案 B 为官方路径。
2. **受限网络**：github release 资产 / ultralytics.com/images 域名被出口代理阻断——资产预置方案见 03_CAPACITY_PLAN §6（SHA256 对账官方值）。
3. **best.pt 文件哈希含时间戳元数据**——跨运行字节不同属预期；复现性以指标/state_dict 论（02_AUDIT §9）。
4. Windows 专用回归（`test_label_path_for_image_windows_separator`）在本机跳过——属设计内行为（skipif nt）。

## 6. 判定

**兼容性评审：PASS。** 双平台隔离干净、文档与实测一致（一处"尚未复验"声明待 Owner 合并后更新）、依赖约束有效、无破坏性变更。
