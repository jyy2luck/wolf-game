# Spec Delta

## MODIFIED Requirements

### Requirement: 生成参数配置
系统 SHALL 支持可选输出预算、重试预算上限、思考模式和 JSON 模式配置；旧三字段配置 SHALL 无需重新引导即可加载，默认输出预算为 4096 tokens、上限为 16384 tokens。未指定思考模式时，DeepSeek 官方服务 SHALL 开启深度思考，其他服务 SHALL 遵循服务默认值；用户明确设置的模式 MUST 优先。系统 SHALL 拒绝非正整数预算、低于初始预算的上限和非法模式值。

#### Scenario: 加载旧配置
- **WHEN** 配置文件仅包含 base_url、api_key、model
- **THEN** 正常加载并采用新生成参数默认值，不修改用户配置文件内容

#### Scenario: 自定义生成参数
- **WHEN** 配置文件包含合法的输出预算、上限、思考模式和 JSON 模式
- **THEN** 游戏请求采用指定参数，配置保存和重载后这些值保持一致

#### Scenario: 非法参数
- **WHEN** 预算不是正整数、上限低于初始预算或模式值非法
- **THEN** 加载时给出明确的配置错误，不发送模型请求

#### Scenario: DeepSeek 默认避免思考耗尽预算
- **WHEN** 使用 DeepSeek 官方服务且 thinking 未指定或为 null
- **THEN** 流式发言与非流式投票均明确发送开启思考参数，并保留有界预算扩容重试

#### Scenario: 用户明确开启思考
- **WHEN** 用户配置 thinking 为 enabled
- **THEN** 请求启用思考并保留有界的预算扩容重试

#### Scenario: 用户明确关闭思考
- **WHEN** 用户配置 thinking 为 disabled
- **THEN** 请求保持关闭思考，不覆盖用户明确配置
