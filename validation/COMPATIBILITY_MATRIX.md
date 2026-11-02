# 跨平台兼容性矩阵(COMPATIBILITY_MATRIX)

- **Windows 原生**:真实执行验证(本轮,物理 GPU)
- **Linux Radeon Cloud**:静态兼容性核查(未在本轮接触真实 Linux GPU 会话)

## 1. 路径与环境假设逐项核查

| 检查点 | Windows 行为(实测) | Linux Radeon Cloud 行为(静态核查) | 处置 |
|---|---|---|---|
| `YOLO_COURSE_HOME` | 优先级最高,正常生效(runner 以默认项目内 `yolo_course` 执行) | 同左 | 无需改动 |
| `/workspace` 检测 | ✅ 已加 POSIX 门控:`Path('/workspace')` 在 Windows 是盘符相对路径,不再可能劫持到 `C:\workspace` | POSIX 分支逐字保留,`/workspace/yolo_course` 行为不变 | lab_utils.root() 1 处最小修改 |
| Notebook SETUP 候选 `[cwd, parent, /workspace/...]` | cwd(项目根)命中,后续候选拦截 | /workspace 候选保留 | 无需改动 |
| 路径分隔符 | 全程 pathlib;data.yaml 内绝对 Windows 路径经 yaml roundtrip 测试通过 | 同左 | 新增测试覆盖 |
| `course` 模块导入 | sys.path 注入项目根,导入成功 | 同左 | 无需改动 |
| 模型下载/缓存 | yolo26n.pt 真实下载(200),统一缓存 `artifacts/yolo26n.pt` + SHA256 | 同左(或课前上传缓存) | 无需改动 |
| COCO128 下载/解压 | 官方 zip 预置缓存路径,解压复用;zip-slip 校验在 Windows 下正常 | 同左 | 无需改动 |
| `data.yaml` 相对/绝对路径 | prepare 写绝对路径;Ultralytics 正常解析 | 同左 | 无需改动 |
| Windows 文件权限 | 项目目录可写,无管理员需求 | N/A | 通过 |
| `workers/num_workers` | 全程 0(原设计),无 spawn 问题 | 0(一致) | 无需改动 |
| Windows 多进程 | 不触发(workers=0;multiprocessing 未用) | N/A | 通过 |
| 临时文件/重命名 | `save_config` 原子替换(tmp→replace)在 NTFS 正常 | 同左 | 通过 |
| Unicode 路径 | 本项目路径 ASCII;代码用 utf-8 显式编码读写 | 同左 | 通过 |
| Kernel 一致性 | 专用 kernel `yolo26-rocm7141`(argv→.venv python),nbclient 强制使用 | 平台 ROCm kernel / 自建同名 kernel | 通过 |
| `/dev/kfd` `/dev/dri` | 仅探测记录(Windows 为 False,预期);require_rocm 检查 `torch.cuda.is_available()`,不依赖设备节点 | Linux 上真实节点检查由平台完成 | 无需改动 |
| matplotlib 无人值守 | runner 设 `MPLBACKEND=module://matplotlib_inline.backend_inline`,inline 渲染无窗口阻塞 | 平台 Jupyter 自带 inline | 通过 |

## 2. 关键平台差异与修复

| 差异 | 影响 | 修复 |
|---|---|---|
| MIOpen hipRTC 无宿主 STL(Windows 运行时栈特有;Linux comgr 注入 STL 不受影响) | Windows 上 Conv/BN 训练 kernel 编译失败 | `HIPRTC_COMPILE_OPTIONS_APPEND` + freestanding shims(Linux 不需要,不激活) |
| `Path('/workspace')` 盘符相对解析 | 潜在输出目录劫持 | POSIX 门控(见上) |
| uv×AMD 索引 `nul` 保留名 | Windows 上 uv lock 失败 | constraints + freeze(见 DEPENDENCY_MANIFEST) |

## 3. build_notebooks.py 重建回归评估

- 本轮对 6 份 Notebook **零修改**,生成器与成品一致;重建不会覆盖任何人工修复。
- 唯一代码修改在 `course/lab_utils.py`(生成器不生成该文件,无覆盖面)。
- 新增回归测试 `tests/test_windows_compat.py::test_all_six_notebooks_present_and_parse`
  持续校验 Notebook 结构完整性(ROCm 门禁、12-Epoch baseline、A/B 评测口径等关键 cell 存在),
  防止未来重建时意外弱化。

## 4. Linux Radeon Cloud 结论

`Linux Radeon Cloud = NOT_RUN / STATIC_COMPATIBILITY_CHECKED`

- POSIX 分支、/workspace 约定、YOLO_USER_DATA_YAML、workers/amp 默认、Notebook 内容均核查兼容;
- 本轮未在真实 Linux GPU 会话执行,**Windows PASS 不代表 Linux 物理 GPU PASS**;
- 上线路径:按 docs/RADEON_CLOUD_LINUX_SETUP.md 在云环境运行
  `scripts/gpu_smoke.py` + `scripts/run_all_labs.py` 取得真实 Linux 证据后方可宣布。
