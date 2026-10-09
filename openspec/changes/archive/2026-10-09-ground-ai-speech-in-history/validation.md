# 验证记录

2026-10-09，变更 `ground-ai-speech-in-history`。

- `.venv/bin/python -m pytest`：170 passed。使用确定性客户端及模拟 HTTP/SSE 响应，无真实模型请求。
- `openspec validate ground-ai-speech-in-history --strict`：通过。
- `git diff --check`：通过。

## 规格与测试对应

| 能力 | 实现与验证 |
| --- | --- |
| game-engine | `SpeechContext` 从实际 speakers 构建；`tests/test_game.py` 覆盖 5 号死亡后的 6、1、2、3、4 顺序、正式记录推进、平安夜、既往放逐、多名死者、跨天重置及完整历史。 |
| ai-players | `tests/test_prompts.py` 覆盖全部五种行动的时间与来源标记、整局记忆及视角隔离；`tests/test_grounding.py` 覆盖真实跨天/已出局玩家引用、编号/座位/时间/原文/正文错配、首位捏造、姓名、轮候、条件假设、主观怀疑及反驳转述（含引号内标点）。 |
| ai-players | `tests/test_agents.py` 与 `tests/test_client.py` 覆盖兼容空依据、统一三次尝试、泄密最多一次纠正、截断预算上限、网络失败安全降级及原始 SSE 内容不进入公开回调。 |
| terminal-ui / game-engine | `tests/test_terminal.py` 串联 Controller、AI、正式事件及真实 rich 渲染，覆盖纠正成功、持续违规兜底、客户端失败、等待清理、正式编号仅分配一次、不重复显示、下一位只读取最终结果、继续投票及新局隔离。 |
| 综合用户场景 | `test_user_scene_then_next_day_valid_human_history_reference` 复现第 1 天 5 号死亡后 6 号捏造 2 号行为被纠正，第 2 天 6 号引用第 1 天人类正式发言 [#2]，确认当天进度重置、旧记录保留、错误候选不传播。 |

## 离线终端演示

运行 `.venv/bin/python main.py --mock --seed 42`。人类为 6 号狼人，输入 `d` 委托队友、发言“我先听大家的意见。”后空行结束，再输入 `0` 弃票。

对局正常结束，11 次离线 LLM 调用。每位 AI 先显示姓名、座位及“生成与校验中”，随后通过事件展示一条带 [#1]–[#4] 的正式正文；人类发言为 [#5]。无候选正文、原始 JSON 或 references 字段展示，等待状态正常清理，继续完成投票和终局。

自然语言检查覆盖明确句式及可核对的引用冲突；任意改写、指代和推理的语义正确性仍超出确定性规则的范围，README 已说明此边界。
