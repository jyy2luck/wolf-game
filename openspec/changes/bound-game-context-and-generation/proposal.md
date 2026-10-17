# Proposal

## Why

长局每次重发全量历史，输入持续增长；深度思考与多层重试还会耗尽输出预算并延长等待。需要在保留真实记录和角色隔离的同时，限制每个行动的上下文、请求次数和总耗时。

## What Changes

- 全局关闭深度思考，兼容已有 enabled 配置。
- 保留完整事件日志，发送有预算的事实、旧发言节选及近期原文，允许按编号有界回查。
- 统一输入预检、输出预算、错误分类、最多三次请求和行动总超时。
- 记录无敏感内容的请求指标，并提供可复现的全 AI 对局验证。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `ai-players`: 有界活跃上下文、原文回查、统一行动预算和请求诊断。
- `config-management`: 输入软预算、可选模型窗口、行动时限以及全局关闭思考。

## Impact

涉及 config.py、llm/client.py、llm/agents.py、新的上下文模块、main.py、测试与自动对局脚本。不增加依赖，不改变引擎完整事件和规则裁决。
