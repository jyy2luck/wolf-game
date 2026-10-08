## Purpose

LLM 狼人杀 MVP 的 `config-management` 能力规格（自 change add-werewolf-mvp 同步）。


### Requirement: 首次启动配置引导
系统在启动时若发现本地不存在 LLM 配置，SHALL 进入交互式引导流程，依次收集 OpenAI 兼容配置项：base_url（提供默认值，可修改）、api_key（输入不回显）、model 名称（提供默认值，可修改）。

#### Scenario: 首次启动无配置
- **WHEN** `~/.llm_murder/config.json` 不存在且用户启动游戏
- **THEN** 系统展示交互式引导，依次询问 base_url、api_key、model，api_key 输入过程中不在终端回显明文

#### Scenario: 用户中断配置
- **WHEN** 用户在引导流程中按下 Ctrl-C 或输入为空且无默认值
- **THEN** 系统退出且不写入任何配置文件

### Requirement: 配置连通性验证
配置引导完成后、写入磁盘前，系统 SHALL 使用所填配置发起一次最小 LLM 请求验证连通性，仅在验证通过后写入配置。

#### Scenario: 配置有效
- **WHEN** 用户填写的 base_url/api_key/model 能成功完成一次最小请求
- **THEN** 系统提示验证通过并将配置写入磁盘

#### Scenario: 配置无效
- **WHEN** 验证请求失败（网络错误、鉴权失败、模型不存在）
- **THEN** 系统展示失败原因并允许用户重新填写，不写入磁盘

### Requirement: 配置安全存储
系统 SHALL 将配置持久化到 `~/.llm_murder/config.json`，文件权限 MUST 为 600（仅文件属主可读写），且配置文件 MUST NOT 写入项目目录。

#### Scenario: 配置落盘
- **WHEN** 配置验证通过被写入
- **THEN** `~/.llm_murder/config.json` 存在且权限为 600，项目目录内不出现任何含 api_key 的文件

### Requirement: 配置自动加载与重配置
系统启动时 SHALL 自动加载已有配置并直接进入游戏；同时 SHALL 提供命令行参数 `--reconfig` 重新进入配置引导。

#### Scenario: 二次启动
- **WHEN** 配置文件已存在且用户正常启动
- **THEN** 系统直接加载配置并开始游戏，不再询问任何配置项

#### Scenario: 手动重置配置
- **WHEN** 用户以 `--reconfig` 参数启动
- **THEN** 系统重新进入交互式配置引导
