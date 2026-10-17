# Validation

验证日期：2026-10-17。

## 自动验证

- `uv run pytest`：559 项通过，包含四种板子的既有流程、终端显示和新增运行日志测试。
- `openspec validate improve-terminal-readability-and-logging --strict`：通过。
- `git diff --check`：通过。
- 日志测试覆盖逐条可读、唯一命名、UTF-8、目录/文件权限、日志配置恢复、现有请求指标、截断/格式错误/鉴权失败、安全降级、异常正文隔离、Ctrl-C、连接关闭异常，以及创建/写入/刷新/关闭失败。
- 日志创建失败的提示在开局清屏后、进入游戏前展示，避免被角色卡清屏擦掉；退出时不会重复提示。

## 离线入口验证

在独立 Python 子进程运行实际 `main.main()`，以临时 `CONFIG_DIR` 隔离本地日志，传入 `--mock --seed 42 --record <临时目录>/game.json`，通过 stdin 提交狼人委托、两轮发言与弃票。

- 正常对局退出码 0，10 条正式发言，公开 JSON 格式版本 2、终局胜负正常，离线请求指标为空。
- 过程没有请求诊断或完整 traceback；终局保留调用、tokens、结构化重试和降级统计，并提示公开记录与诊断日志路径。
- 独立运行在首次狼人输入提示处发送 SIGINT：退出码 130，之前的日志及 `status=interrupted` 保留，终端提示日志位置。
- 正常日志记录 `status=completed`；两个子进程使用不同日志文件。

本机临时证据目录：`/var/folders/x_/xxgzpyt10tz_nvgtb_qq7npc0000gn/T/wolf-terminal-validation-_uqvcqms/`。其中 `normal/` 与 `interrupted/` 分别保存 stdout、stderr 和日志；`normal/game.json` 为公开记录。

## 显示检查

- 从实际正式事件渲染路径捕获 Rich 输出：标题使用加粗 cyan；正文不包含额外 ANSI 样式，引用和多行原文保留，发言块间存在空行。
- 保存同一输出的无颜色文本，以及深色 MONOKAI、浅色默认终端主题的 SVG/PNG 预览，逐图检查标题、正文对比和间距。
- Rich 的 SVG 导出按字符串长度计算中文 `textLength`，预览时将其修正为 Rich 的终端单元格宽度，避免 SVG 中文重叠。此修正仅作用于临时预览文件，应用仍直接输出终端文本。
- 禁色输出保留编号、姓名、座位、发言窗口和段落层次；实际终端尊重用户自己的前景色与禁色设置。

本次未调用真实 API，也未逐一检查用户的终端字体与 ANSI 调色板；请求协议和生成预算未改动。日志按局保留，需按 README 说明手动清理。
