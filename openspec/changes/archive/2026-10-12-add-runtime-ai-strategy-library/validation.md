# 验证记录

2026-10-11，变更 `add-runtime-ai-strategy-library`。

- `uv run pytest tests/test_strategies.py`：38 passed。
- `uv run pytest tests/test_agents.py tests/test_prompts.py tests/test_terminal.py`：91 passed。
- `uv run pytest`：286 passed，2 failed。失败为既有 Windows 权限断言：`test_old_config_uses_new_defaults_without_rewriting`、`test_generation_options_roundtrip` 期望 POSIX `0600`，当前平台返回 `0666`。本变更未修改 `config.py` 或 `tests/test_config.py`。
- `uv build --wheel`：构建成功。在仓库外的隔离环境安装 wheel 后，包资源加载得到 67 条策略、10 类身份，并成功执行实时检索；没有依赖仓库中的 JSON。
- `openspec validate add-runtime-ai-strategy-library --strict`：通过。
- `git diff --check`：通过。

## 覆盖范围

| 验证内容 | 证据 |
| --- | --- |
| 库完整性 | 67 个策略编号与设计对应；10 类身份各有七档连续、不重叠的人数指导；重复编号、缺少基础卡和未定义身份会被拒绝。 |
| 信息隔离和不确定性 | 假查验及他人药量声明不成为事实；自身私有查验、刀口及队友信息只用于合法角色；未知狼数保留候选；矛盾信息降级。 |
| 规则和技能时序 | 非普通人数规则及未完成结算不套用普通判胜公式；猎人策略依赖明确能力；有死亡反击时不启用普通必毒卡。 |
| 每次重新检索 | 覆盖首夜、真实查杀、解药耗尽、本人被刀和人数变化；实际 AIPlayer 的女巫跨轮请求按资源变化切换策略，每次仍只有原有一次模型请求。 |
| 五个入口及兼容性 | 发言、投票、狼刀、查验、用药均包含参考；夜间剔除死亡狼队友；既有输出解析、重试、可见历史及终端回归通过。 |
| 有界参考和失败降级 | 最多 8 张卡、4000 字符；当前各行动保留基础判断卡；文件缺失或校验失败仍能生成原有行动请求。 |

本次使用离线及确定性客户端验证，没有向真实模型服务发起请求。验证说明检索、接入与打包可用，不代表实战胜率或策略质量已经得到证明。当前引擎仍为四身份六人局；猎人等扩展玩法需另行实现角色技能和结算规则。
