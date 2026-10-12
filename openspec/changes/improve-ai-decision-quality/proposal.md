# Proposal

## Why

AI 能读取历史和策略，却漏掉确定性的残局结论、沿用过时判断并选择不适用的打法，实际发言和决策缺少贡献。默认关闭思考、狼队只协调刀口以及每天仅一轮发言进一步削弱博弈体验。

## What Changes

- DeepSeek 默认开启思考，保留明确配置及截断预算重试。
- 从合法硬事实闭合阵营排除推理，提示具体人数危险和行动后果。
- 发言、投票和技能在原有请求中更新私有摘要，按行动降低决策随机性。
- 根据有出处的身份声明、对跳及被查杀场景匹配策略；明确发言任务并适度放宽长度。
- 狼窝原有请求同时生成共享白天计划，严格限制狼队可见。
- 增加一轮简短回应，投票结束后公布个人票型并核对相关发言。
- 提供公开对局记录、重试及降级统计，补充固定局面回归验证。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `ai-players`: 连续决策记忆、行动任务、狼队计划及公开票型核对。
- `ai-strategy-library`: 硬事实推理闭合和声明场景筛选。
- `config-management`: 默认开启 DeepSeek 思考。
- `game-engine`: 两轮白天交流、个人票型及狼队计划事件。
- `terminal-ui`: 回应提示、决策统计和可选公开对局记录。

## Impact

涉及 llm/agents.py、llm/strategies.py、策略数据、llm/client.py、engine/events.py、engine/game.py、main.py、ui/terminal.py、对应测试与 README。不增加依赖；旧 JSON 响应继续可解析，旧配置无需改写。回应阶段会增加每个存活 AI 每天一次调用，思考模式会增加延迟及 token 消耗。
