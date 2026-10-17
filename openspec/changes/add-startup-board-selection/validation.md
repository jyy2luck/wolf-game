# Validation

- `uv run pytest`：581 passed，9.48 秒，无警告。包括新增 22 项启动相关用例、既有四种板子的完整 Controller mock 对局，以及终局统计、日志和公开记录回归。
- `openspec validate add-startup-board-selection --strict`：通过。
- `git diff --check`：通过。
- `uv run python main.py --help`：列出全部四种板子，并说明未传参数显示菜单、显式指定跳过菜单及回车默认六人局。
- 80 列无颜色菜单检查：四种玩法名称、人数、完整角色配置及警长开关均可读，两种十二人局的白痴/守卫配置完整换行显示。
- 实际子进程离线冒烟 10 个场景：菜单编号 1/2/3/4、菜单回车默认、四种显式 --board 和未选择时 EOF。选定板子的 9 个场景均进入对应角色卡并记录正确板子，显式参数未展示菜单；所有场景在输入结束后以 130 安全退出。菜单直接 EOF 未建局。
- 冒烟使用临时 CONFIG_DIR 与 --mock，不访问用户 LLM 配置或真实模型。终端输出、诊断日志和结果保存在临时目录 `/var/folders/x_/xxgzpyt10tz_nvgtb_qq7npc0000gn/T/wolf-board-selection-wx4ymdk4`；不将这些运行产物加入仓库。

本次未调用真实付费模型；真实模型的对局质量与调用成本不在本次启动入口验证范围内。
