# Spec Delta

## MODIFIED Requirements

### Requirement: 配置自动加载与重配置
系统 SHALL 在板子确定后自动加载已有配置并进入游戏；同时 SHALL 提供命令行参数 `--reconfig` 重新进入配置引导。未指定 `--board` 时 SHALL 先完成板子选择，再加载配置或进入首次/重新配置引导；离线模式 SHALL 同样支持板子选择并保持无需 LLM 配置。

#### Scenario: 二次启动
- **WHEN** 配置文件已存在且用户正常启动
- **THEN** 系统在板子确定后直接加载配置并开始游戏，不再询问任何配置项

#### Scenario: 手动重置配置
- **WHEN** 用户以 `--reconfig` 参数启动
- **THEN** 系统在板子确定后重新进入交互式配置引导

#### Scenario: 首次启动先选板子
- **WHEN** 配置文件不存在且未指定 --board
- **THEN** 玩家先选择板子，选定后再填写并验证 LLM 连接配置

#### Scenario: 离线启动选择板子
- **WHEN** 用户使用 --mock 且未指定 --board
- **THEN** 玩家先选择板子，随后直接创建该板子的离线对局，不加载或验证真实 LLM 配置
