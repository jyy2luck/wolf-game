"""基于公开原始事件核对引用，检查有限的明确发言行为断言。

不以玩家转述构造被指控者的记录，也不尝试证明任意中文推理的语义。
"""

from collections.abc import Sequence
import re

from pydantic import BaseModel, Field

from engine.events import Event, EventKind
from engine.game import SpeechContext


class SpeechReference(BaseModel):
    speech_index: int = Field(ge=1, strict=True)
    seat: int = Field(ge=1, le=6, strict=True)
    day: int = Field(ge=1, strict=True)
    quote: str = Field(min_length=1)


_CITATION = re.compile(r"\[#(\d+)\]")
# 引号内的标点不切断转述范围，防止把被反驳的指控当作本人断言。
_CLAUSE = re.compile(r'(?:“[^”]*”|「[^」]*」|『[^』]*』|‘[^’]*’|"[^"]*"|[^，,。！？!?；;\n])+')
_QUOTED = re.compile(r'“([^”]*)”|「([^」]*)」|『([^』]*)』|‘([^’]*)’|"([^"]*)"')
_REPORT = re.compile(r"声称|宣称|说过|说了|说|表示|提到")
# shortcut: 只核对明确句式；遇到可复现的漏检或误伤时补充成对用例。
_BEHAVIOR = re.compile(
    r"(?:一上来就)?急着说话|抢(?:着)?(?:说话|发言|话)|改口|"
    r"长篇大论|(?:一直|不停|反复)(?:在)?(?:给自己|为自己)?辩解|"
    r"发言(?:太|很|这么|过于)?(?:积极|激进|急|主动)|"
    r"(?:刚才|刚刚|之前|先前|已经|一上来)(?:就)?说|说(?:了|过)|声称|宣称|表示|提到"
)
_OBSERVATION = re.compile(r"眼神(?:明显|有些|很)?躲闪|拍(?:了)?桌(?:子)?|表情(?:很|明显)?慌张")
_DEATH = re.compile(r"死了|死亡|被(?:放逐|投出|票出)(?:了|出局)?|出局了")
_VOTE_COUNT = re.compile(r"(?:得(?:了)?|获(?:得)?(?:了)?|有)?\s*(\d+)\s*票")
_PERSONAL_VOTE = re.compile(
    r"^\s*(?:(?:今天|今日|昨天|昨日|前天|第\s*\d+\s*天)\s*)?(?:把票)?"
    r"(?:投(?:票)?(?:给|向)?(?:了)?\s*([1-6])\s*号|弃(?:了)?票)"
)
_PREDICTED = re.compile(r"将|会|可能|大概|预计|估计|应该")
_SILENCE = re.compile(r"(?:故意|刻意)(?:不说话|不发言|沉默)|(?:躲|逃避)(?:着)?发言")
_FUTURE = re.compile(r"如果|假如|假设|要是|倘若|等(?:待|到|听)?|发言后|说话后")
_NEGATIVE = re.compile(r"(?:没有|没|未|不曾|不是|并非|并不|尚无)[^，,。！？!?；;]{0,8}$")
_DENIED_PREFIX = re.compile(r"(?:不能|无法|并未|未能)(?:证明|说明|认定)|不(?:能)?代表|(?:没有|尚无)(?:记录|证据)(?:能)?证明")
_TIME = re.compile(r"第\s*(\d+)\s*天|今天|今日|刚才|刚刚|昨天|昨日|前天|昨晚|昨夜")


def speech_index(view: Sequence[Event]) -> dict[int, Event]:
    """整局可见的公开发言，不按今天或存活状态裁剪。"""
    return {
        e.speech_index: e for e in view
        if e.public and e.kind is EventKind.SPEECH and e.speech_index is not None
    }


def _mentions(text: str, names: dict[int, str]) -> list[tuple[int, int, int]]:
    mentions = [(m.start(), m.end(), int(m[1]) - 1)
                for m in re.finditer(r"(?<!\d)([1-6])\s*号", text)]
    for seat, name in names.items():
        # 单字人称（例如座位表中的“你”）不能可靠地当作姓名解析。
        if len(name) < 2:
            continue
        mentions.extend((m.start(), m.end(), seat) for m in re.finditer(re.escape(name), text))
    return sorted(mentions)


def _day_at(text: str, current_day: int, end: int | None = None) -> int | None:
    times = list(_TIME.finditer(text if end is None else text[:end]))
    if not times:
        return None
    time = times[-1]
    if time[1]:
        return int(time[1])
    return current_day - (2 if time[0] == "前天" else 1 if time[0] in ("昨天", "昨日") else 0)


def _mask_reports(text: str, records: dict[int, Event], names: dict[int, str]) -> str:
    """仅屏蔽有真实引用、原文吻合的转述范围；保留外层发言人用于校验。"""
    spans = []
    for clause_match in _CLAUSE.finditer(text):
        clause = clause_match[0]
        cited = [records[int(m[1])] for m in _CITATION.finditer(clause) if int(m[1]) in records]
        for report in _REPORT.finditer(clause):
            authors = [m for m in _mentions(clause[:report.start()], names)]
            if not authors:
                continue
            author = authors[-1][2]
            sources = [e for e in cited if e.seat == author]
            if not sources:
                continue
            tail = clause[report.end():]
            quoted = _QUOTED.search(tail)
            if quoted:
                content = next(g for g in quoted.groups() if g is not None)
                start, end = report.end() + quoted.start(), report.end() + quoted.end()
            else:
                content = _CITATION.sub("", tail).strip(" ：:说了过")
                start, end = report.end(), len(clause)
            if content and any(content in e.extra["content"] for e in sources):
                spans.append((clause_match.start() + start, clause_match.start() + end))
    chars = list(text)
    for start, end in spans:
        # 保留编号，使外层行为仍必须绑定来源。
        citations = {i for m in _CITATION.finditer(text, start, end) for i in range(m.start(), m.end())}
        for i in range(start, end):
            if i not in citations:
                chars[i] = " "
    return "".join(chars)


def validate_speech_facts(
    speech: str,
    references: Sequence[SpeechReference],
    *,
    records: dict[int, Event],
    day: int,
    context: SpeechContext | None = None,
    view: Sequence[Event] | None = None,
) -> str | None:
    body_ids = {int(m[1]) for m in _CITATION.finditer(speech)}
    refs = {r.speech_index: r for r in references}
    if len(refs) != len(references):
        return "引用依据的 speech_index 重复"
    if missing := body_ids - refs.keys():
        return f"正文引用 {sorted(missing)} 缺少 references 依据"
    if unused := refs.keys() - body_ids:
        return f"references 编号 {sorted(unused)} 未在正文使用"
    for index, ref in refs.items():
        event = records.get(index)
        if event is None:
            return f"[#{index}] 不存在于整局公开发言记录"
        if ref.seat != event.seat + 1:
            return f"[#{index}] 发言人是 {event.seat + 1} 号，不是 {ref.seat} 号"
        if ref.day != event.day:
            return f"[#{index}] 属于第 {event.day} 天，不是第 {ref.day} 天"
        if not ref.quote.strip() or ref.quote not in event.extra["content"]:
            return f"[#{index}] quote 不是该玩家发言原文片段"

    names = {e.seat: e.extra["name"] for e in records.values()}
    if context is not None:
        names.update(dict(context.names))
    masked = _mask_reports(speech, records, names)
    for original, clause_match in zip(_CLAUSE.finditer(speech), _CLAUSE.finditer(masked)):
        clause = clause_match[0]
        mentions = _mentions(clause, names)
        citations = list(_CITATION.finditer(clause))
        for citation in citations:
            event = records[int(citation[1])]
            before = [m for m in mentions if m[0] < citation.start()]
            # 编号紧接人物或人物行为，视为明确来源关联。
            associated = before[-1] if before else next(iter(mentions), None)
            if associated is not None and associated[2] != event.seat:
                return f"正文 [#{event.speech_index}] 关联 {associated[2] + 1} 号，但记录发言人是 {event.seat + 1} 号"
            expected_day = _day_at(clause, day, citation.start())
            if expected_day is None:
                expected_day = _day_at(clause, day)
            if expected_day is not None and expected_day != event.day:
                return f"[#{event.speech_index}] 属于第 {event.day} 天，正文时间指向第 {expected_day} 天；请使用正确历史时间"
            if _REPORT.search(original[0]):
                for quotation in _QUOTED.finditer(original[0]):
                    quote = next(g for g in quotation.groups() if g is not None)
                    if quote and quote not in event.extra["content"]:
                        return f"正文 [#{event.speech_index}] 引号内文字不是对应原文"

        for i, (start, end, seat) in enumerate(mentions):
            prefix = clause[:start]
            personal_vote = _PERSONAL_VOTE.match(clause[end:]) if view is not None else None
            if personal_vote is not None:
                statement = prefix + clause[end:end + personal_vote.end()]
                if _FUTURE.search(statement) or _PREDICTED.search(statement) or _DENIED_PREFIX.search(prefix):
                    continue
                requested_day = _day_at(statement, day)
                results = [e for e in view if e.public and e.kind is EventKind.VOTE_RESULT
                           and (requested_day is None or e.day == requested_day)]
                if not results or "votes" not in results[-1].extra:
                    return "所述时间尚无公开个人票型，不能编造个人投票目标"
                result = results[-1]
                votes = dict(result.extra["votes"])
                target = int(personal_vote[1]) - 1 if personal_vote[1] else None
                if seat not in votes or votes[seat] != target:
                    actual = (f"投给{votes[seat] + 1}号" if seat in votes and votes[seat] is not None
                              else "弃票" if seat in votes else "没有个人投票记录")
                    return f"第{result.day}天公开个人票型中{seat + 1}号{actual}，与正文不符"
                continue
            tail = clause[end:mentions[i + 1][0] if i + 1 < len(mentions) else len(clause)]
            observation = _OBSERVATION.search(tail)
            death = _DEATH.search(tail) if view is not None else None
            votes = _VOTE_COUNT.search(tail) if view is not None else None
            silence = _SILENCE.search(tail)
            action = min((m for m in (observation, death, votes, _BEHAVIOR.search(tail), silence)
                          if m is not None), key=lambda m: m.start(), default=None)
            if action is None or _FUTURE.search(prefix + tail[:action.end()]):
                continue
            if _DENIED_PREFIX.search(prefix):
                continue
            if _NEGATIVE.search(tail[:action.start()]):
                continue
            if (action is death or action is votes) and _PREDICTED.search(prefix + tail[:action.start()]):
                continue
            requested_day = _day_at(prefix + tail[:action.end()], day)
            if action is observation:
                return f"文字事件没有 {seat + 1} 号「{observation[0]}」的观察依据"
            if action is death:
                deaths = [e for e in view if e.public and (
                    (e.kind is EventKind.DEATH and seat in e.extra['seats'])
                    or (e.kind is EventKind.EXILE and e.seat == seat)
                )]
                if death[0].startswith("被"):
                    deaths = [e for e in deaths if e.kind is EventKind.EXILE]
                if not any(requested_day is None or e.day == requested_day for e in deaths):
                    return f"公开记录没有 {seat + 1} 号在所述时间「{death[0]}」的事件"
                continue
            if action is votes:
                results = [e for e in view if e.public and e.kind is EventKind.VOTE_RESULT
                           and (requested_day is None or e.day == requested_day)]
                if not results:
                    return "所述时间尚无公开票型，不能编造得票数"
                result = results[-1]
                actual = result.extra['tally'].get(seat, 0)
                if int(votes[1]) != actual:
                    return f"第 {result.day} 天公开票型中 {seat + 1} 号是 {actual} 票，不是 {votes[1]} 票"
                continue
            player_records = [e for e in records.values() if e.seat == seat]
            if action is silence:
                if (context is not None and seat in context.remaining) or not any(e.day == day for e in player_records):
                    return f"{seat + 1} 号今天尚未完成发言，正常轮候不能作为故意沉默或躲发言的证据"
            elif not player_records:
                return f"{seat + 1} 号整局尚无发言记录，不能断言其「{action[0]}」"
            elif requested_day is not None and not any(e.day == requested_day for e in player_records):
                return f"{seat + 1} 号第 {requested_day} 天尚无发言记录；往日记录不能写成今天刚才"
            elif not any(records[int(c[1])].seat == seat for c in citations):
                return f"对 {seat + 1} 号已发生发言行为的描述缺少真实编号及 references"
    return None
