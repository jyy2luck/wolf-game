## MODIFIED Requirements

### Requirement: 生成参数配置
系统 SHALL 支持可选输出预算、重试预算上限、思考模式和 JSON 模式配置；旧三字段配置 SHALL 无需重新引导即可加载，默认输出预算为 4096 tokens、上限为 16384 tokens。所有游戏请求 SHALL 全局关闭深度思考，包括已有 enabled 配置；DeepSeek 或显式配置 thinking 的兼容端点 SHALL 发送 disabled，其他服务 SHALL 不发送未知扩展参数。系统 SHALL 拒绝非正整数预算、低于初始预算的上限和非法模式值。

#### Scenario: 加载旧配置
- **WHEN** 配置文件仅包含 base_url、api_key、model
- **THEN** 正常加载并采用新生成参数默认值，不修改用户配置文件内容

#### Scenario: 自定义生成参数
- **WHEN** 配置文件包含合法的输出预算、上限、思考模式和 JSON 模式
- **THEN** 游戏请求采用指定预算和 JSON 参数，thinking 受全局关闭策略约束，配置保存和重载后这些值保持一致

#### Scenario: 非法参数
- **WHEN** 预算不是正整数、上限低于初始预算或模式值非法
- **THEN** 加载时给出明确的配置错误，不发送模型请求

#### Scenario: DeepSeek 默认避免思考耗尽预算
- **WHEN** 使用 DeepSeek 官方服务且 thinking 未指定或为 null
- **THEN** 流式发言与非流式投票均明确发送关闭思考参数

#### Scenario: 用户明确开启思考
- **WHEN** 用户配置 thinking 为 enabled
- **THEN** 运行时仍关闭思考并保留有界的预算扩容重试，不重写旧配置

## ADDED Requirements

### Requirement: 输入及行动预算配置
系统 SHALL 支持正整数输入软预算（默认 12000 tokens）、可选的经核实模型窗口和正数行动总时限（默认 90 秒）。预算估算 MUST 预留输出及安全余量，非法或无法容纳输入的配置 SHALL 明确拒绝；缺少新字段的旧配置 SHALL 使用默认值。

#### Scenario: 未配置模型窗口
- **WHEN** 使用旧配置
- **THEN** 输入软预算生效且不假定模型真实容量

#### Scenario: 配置已知模型窗口
- **WHEN** 配置合法模型窗口
- **THEN** 输入上限不超过窗口减去本次输出预算和安全余量
