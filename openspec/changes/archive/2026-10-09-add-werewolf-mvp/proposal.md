## Why

想玩一局随时能开、AI 演技在线的狼人杀：无需组人、无需联网对战平台，一个人打开终端即可与 5 个 LLM 扮演的玩家对局。市面上缺少把"确定性规则引擎 + LLM 角色扮演"结合好的单机狼人杀，这也是验证"LLM 社交推理游戏"可行性的最小实验场。

## What Changes

- 新增终端 CLI 狼人杀游戏（Python + rich），6 人局：狼人 ×2、预言家、女巫、平民 ×2
- 新增启动时 LLM 配置流程：首次启动交互式引导填写 OpenAI 兼容配置（base_url / api_key / model），持久化到本地 `~/.llm_murder/config.json`（权限 600），之后启动自动加载
- 玩家随机获得一个身份，其余 5 个角色由 AI 扮演；AI 与玩家遵循同一套视角隔离机制
- 新增确定性游戏引擎：夜晚（狼刀→预言家查验→女巫用药）→ 白天轮流发言 → 统一投票 → 胜负判定；规则判定全部由代码完成，LLM 不参与规则裁决
- 新增 AI 玩家系统：按角色过滤的信息视角 + 人设卡 + 发言编号引用 + 穿帮检测器（规则匹配泄密重试）+ 发言/投票同包结构化输出
- 白天发言采用流式输出到终端，营造真实对话节奏
- 第一版从简：不设警长/警徽流程；平票视为无人出局

## Capabilities

### New Capabilities

- `config-management`: LLM 连接配置的引导、校验、持久化与加载（OpenAI 兼容 base_url/api_key/model）
- `game-engine`: 确定性对局状态机——角色分配、昼夜循环、技能结算、投票计数、胜负判定与视角隔离
- `ai-players`: AI 玩家的 prompt 拼装、人设系统、记忆与一致性、结构化决策输出、穿帮检测与重试
- `terminal-ui`: 终端交互界面——发言流式渲染、玩家输入（自由发言/技能选择/投票）、夜晚遮罩与角色提示

### Modified Capabilities

（无 —— 全新项目，不修改既有规格）

## Impact

- 全新代码库：`main.py`、`config.py`、`engine/`（game/roles/events）、`llm/`（client/agents）、`ui/terminal.py`
- 新增依赖：`openai`（OpenAI 兼容客户端）、`rich`（终端 UI）、`pydantic`（结构化输出校验）
- 外部依赖：任一 OpenAI 兼容 LLM 服务（DeepSeek/Qwen/OpenAI 等），需用户自备 api_key
- 本地数据：`~/.llm_murder/config.json` 存放敏感 api_key，需严格权限控制；对局日志可选留存
- 成本特征：一局约 60~100 次 LLM 调用，调用基本串行（白天发言依赖前序发言），单局 10~20 分钟
