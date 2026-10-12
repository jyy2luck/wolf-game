# Spec Delta

## MODIFIED Requirements

### Requirement: 生成参数配置
系统 SHALL 支持可选输出基础预算、预算上限、思考模式和 JSON 模式配置；旧三字段配置 SHALL 无需重新引导即可加载，默认基础预算为 4096 tokens、上限为 16384 tokens。未指定思考模式时，DeepSeek 官方服务 SHALL 开启深度思考，其他服务 SHALL 遵循服务默认值；用户明确设置的模式 MUST 优先。明确开启思考的游戏请求 SHALL 将起始预算提高到基础预算与 8192 的较大值，但 MUST NOT 超过用户预算上限；其他模式 SHALL 从基础预算开始。截断扩大预算后，同一客户端后续游戏行动 SHALL 复用扩大后的预算，MUST NOT 改写配置；新局新客户端 SHALL 重置。系统 SHALL 拒绝非正整数预算、低于基础预算的上限和非法模式值。

#### Scenario: 加载旧配置
- **WHEN** 配置文件仅包含 base_url、api_key、model
- **THEN** 正常加载并采用新生成参数默认值，不修改用户配置文件内容

#### Scenario: 自定义生成参数
- **WHEN** 配置文件包含合法的输出预算、上限、思考模式和 JSON 模式
- **THEN** 游戏请求根据指定基础预算、上限和模式计算运行时预算，配置保存和重载后这些值保持一致

#### Scenario: 非法参数
- **WHEN** 预算不是正整数、上限低于初始预算或模式值非法
- **THEN** 加载时给出明确的配置错误，不发送模型请求

#### Scenario: DeepSeek 默认避免思考耗尽预算
- **WHEN** 使用 DeepSeek 官方服务且 thinking 未指定或为 null
- **THEN** 流式发言与非流式投票均明确发送开启思考参数，默认从 8192 tokens 开始，并保留有界预算扩容重试

#### Scenario: 用户明确开启思考
- **WHEN** 用户配置 thinking 为 enabled
- **THEN** 请求启用思考并保留有界的预算扩容重试

#### Scenario: 用户明确关闭思考
- **WHEN** 用户配置 thinking 为 disabled
- **THEN** 请求保持关闭思考，不覆盖用户明确配置

#### Scenario: 预算上限低于思考起点
- **WHEN** 基础预算为 4096、上限为 6000 且思考开启
- **THEN** 请求从 6000 tokens 开始，不超过用户上限，不能继续扩容时停止无效重试

#### Scenario: 后续行动复用扩容预算
- **WHEN** 一个游戏行动从 8192 扩容到 16384 后完成，下一个玩家开始行动
- **THEN** 下一个行动直接使用 16384，配置仍保持原始值，创建新局客户端时恢复初始运行时预算
