# Proposal

## Why

DeepSeek Flash 默认高强度思考会在投票中耗尽 4096 或 8192 tokens，正文仍为空。单纯扩大预算让每轮投票等待数分钟，因此需要有效的默认生成策略及有界的截断重试。

## What Changes

- DeepSeek 官方服务默认关闭深度思考，允许用户明确开启；支持配置预算、预算上限及 JSON 模式，兼容已有三字段配置。
- 流式与非流式调用识别输出截断；结构化调用在有限重试次数内扩大预算，记录完成原因和 token 统计。
- 为支持的服务启用 JSON 输出，所有行动提示词使用合法 JSON 示例。
- 补充真实 SDK 响应形状的回归测试和配置说明。
- 为各行动提供本局固定规则，修复人类玩家不在 1 号时名字为空的问题，统计并展示思考 token 用量。

## Capabilities

### New Capabilities

### Modified Capabilities

- `ai-players`: 区分输出截断与格式错误，按原因重试并在预算及次数上限内降级。
- `config-management`: 支持可选生成参数，旧配置自动采用安全默认值。

## Impact

涉及 `config.py`、`llm/client.py`、`llm/agents.py`、`main.py`、测试和 README。保持确定性游戏规则与现有配置存储位置；无新增依赖。
