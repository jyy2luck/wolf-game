## MODIFIED Requirements

### Requirement: 解析容错与降级
AI 结构化输出 SHALL 经 schema 校验：解析失败时携带错误提示重试（最多 2 次）；输出因长度限制截断时 SHALL 区分于格式错误，在配置的输出预算上限内增加预算重试，且不将截断文本追加至对话。仍失败则降级为预设安全行为（发言轮替换为保留意见发言，投票轮弃票），游戏 MUST NOT 因 LLM 输出异常而中断。

#### Scenario: 输出可解析
- **WHEN** AI 返回的完整文本经提取后通过 schema 校验
- **THEN** 正常采用其发言与倾向

#### Scenario: 持续解析失败
- **WHEN** 同一 AI 连续 3 次输出均无法解析
- **THEN** 该次发言以安全发言占位或该次投票记为弃票，对局继续

#### Scenario: 思考耗尽输出预算
- **WHEN** 流式或非流式请求因长度限制结束，正文为空或 JSON 不完整
- **THEN** 系统记录截断原因与 token 统计，在预算上限内扩大预算重新请求，成功后采用完整输出

#### Scenario: 达到重试或预算上限
- **WHEN** 截断请求已用尽最多 2 次重试或已经达到预算上限
- **THEN** 系统停止请求并采用该行动的预设安全行为，不无限重试

#### Scenario: 网络重试耗尽
- **WHEN** 模型请求的网络重试全部失败
- **THEN** 该次行动使用预设安全行为，对局继续

## ADDED Requirements

### Requirement: JSON 输出约束
所有行动的提示词 SHALL 提供合法 JSON 示例。DeepSeek 官方服务 SHALL 默认启用 JSON 输出模式，其他兼容服务 SHALL 仅在明确配置支持时启用该模式，并继续进行 schema 与行动目标校验。

#### Scenario: JSON 模式兼容
- **WHEN** 使用其他 OpenAI 兼容服务且没有指定 JSON 模式
- **THEN** 请求不携带 JSON 模式参数，仍使用合法示例和解析容错

#### Scenario: 明确关闭 JSON 模式
- **WHEN** 用户将 JSON 模式配置为关闭
- **THEN** 请求不携带 JSON 模式参数

### Requirement: 固定规则上下文
所有 AI 行动的上下文 SHALL 包含本局固定角色配置与规则，包括没有守卫或其他保护角色，避免将不存在的角色作为推理依据。

#### Scenario: 平安夜分析
- **WHEN** AI 收到平安夜事件并进行发言或投票
- **THEN** 其上下文明确本局只有两狼、一预言家、一女巫及两平民，没有守卫
