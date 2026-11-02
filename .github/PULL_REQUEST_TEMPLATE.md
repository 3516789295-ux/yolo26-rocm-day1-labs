<!-- 中文模板在前 · English template below -->

## 变更说明 | What & Why

<!-- 这个 PR 做了什么?为什么?关联 Issue(#编号) -->
<!-- What does this PR change and why? Link related issues. -->

## 类型 | Type

- [ ] 📖 文档/翻译 | Docs / translation
- [ ] 🐛 Bug 修复 | Bug fix
- [ ] ✨ 新实验/新检查 | New experiment / new check
- [ ] 🔧 环境/工具链 | Environment / tooling

## 验证证据 | Verification Evidence

<!-- 粘贴 pytest / pip check / notebook 执行结果 -->
<!-- Paste pytest / pip check / notebook run results. -->

```
pytest: 
pip check: 
```

## 诚信检查 | Integrity Checklist

- [ ] 未删除/绕过 ROCm 门禁、数据泄漏检测或 A/B 纯度断言 | No ROCm gate / leak check / A-B purity assertion removed or bypassed
- [ ] 未预置性能或精度结论 | No pre-baked performance or accuracy conclusions
- [ ] 未提交权重/数据集/敏感信息 | No weights, datasets, or sensitive data committed
- [ ] Notebook 改动通过 `build_notebooks.py` 生成且保持中英对照 | Notebook edits generated via `build_notebooks.py`, still bilingual
