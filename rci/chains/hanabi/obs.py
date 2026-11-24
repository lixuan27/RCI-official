"""What one seat sees: its projection of a state, the observation text, the legal codes and the system prompt.

The text holds the other players' hands, every card's still-possible colours and ranks after all hints (the observer's
own cards included), fireworks, tokens, lives, deck size, the discard pile, the last eight moves and the legal moves,
plus marks derived from public cards only (! playable now, x never playable, * last copy still needed). `render`
takes a View, which holds no card of the observer's own hand and not the deck order.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import NamedTuple, Optional, Tuple

from rci.chains.hanabi import engine as E

LAST_N = 8                    # moves shown under "Last actions"

# The system prompt: identical for every seat, turn, generation and chain.
SYSTEM = (
    "Hanabi, standard rules: four players on one team sharing one score.\n"
    "Cards: colors R Y G W B, ranks 1-5; per color three 1s, two each of 2-4, one 5 (50 cards). Each player starts "
    "with 4 cards and sees every hand but their own.\n"
    "Each color's firework is built 1 to 5 in order; a card is playable if its rank is one above its firework's top.\n"
    "The team starts with 8 hint tokens and 3 lives. One action per turn:\n"
    "P<slot> play. Unplayable: discarded, lose a life. Completing a firework with its 5 returns a token (max 8).\n"
    "D<slot> discard, regain a token; not allowed at 8 tokens.\n"
    "H<k><color or rank> spend a token to show player P+k all their cards of that color or rank (at least one).\n"
    "After playing or discarding, draw if the deck is not empty. Losing the 3rd life ends the game with score 0. After "
    "the last card is drawn, everyone (the drawer too) takes one more turn. Score = cards in fireworks (max 25).\n"
    "State: slots 1-4 oldest to newest; a card leaving moves later ones up, the new card goes last. P+k acts k turns "
    "after you. [colors|ranks] = what that card's holder can still believe it is after every hint, including what "
    "hints said it is not; ~ = touched by a hint. Marks: ! playable now; x never playable; * last copy, still needed. "
    "(slots) after a hint code = the slots it touches.\n"
    "Reply with one legal action code only."
)
SYSTEM_SHA256 = hashlib.sha256(SYSTEM.encode()).hexdigest()


class View(NamedTuple):
    """One seat's projection of a state: no own card, no deck order."""
    seat: int
    turn: int
    deck: int
    ttp: int
    info: int
    lives: int
    fw: Tuple[int, ...]
    discard: Tuple[E.Card, ...]                     # sorted by color then rank
    own: Tuple[Tuple[int, int, bool], ...]          # (colors mask, ranks mask, touched) a slot
    others: Tuple[Tuple[E.HandCard, ...], ...]      # P+1, P+2, P+3
    hist: Tuple[E.Event, ...]                       # the last LAST_N moves
    end: Optional[str]


def project(s: E.State, seat: int) -> View:
    own = tuple((hc.cm, hc.rm, hc.touched) for hc in s.hands[seat])
    others = tuple(s.hands[(seat + k) % E.PLAYERS] for k in range(1, E.PLAYERS))
    return View(seat, s.turn, s.deck, s.ttp, s.info, s.lives, tuple(s.fw), tuple(sorted(s.discard)), own, others,
                tuple(s.hist[-LAST_N:]), s.end)


# ------------------------------------------------------------------------------------------------- marks
def dead(c: int, r: int, fw, dcount) -> bool:
    """Never playable: its firework is past it, or a lower rank of its color between the top and it is all discarded."""
    if r <= fw[c]:
        return True
    return any(dcount.get((c, r2), 0) >= E.COPIES[r2] for r2 in range(fw[c] + 1, r))


def marks(card: E.Card, fw, dcount) -> str:
    """'' | '!' | 'x' | '*' | '!*' for a visible card, from the fireworks and the discard pile only."""
    c, r = card
    if dead(c, r, fw, dcount):
        return "x"
    m = "!" if fw[c] == r - 1 else ""
    if dcount.get((c, r), 0) == E.COPIES[r] - 1:
        m += "*"
    return m


def own_mark(cm: int, rm: int, fw, dcount) -> str:
    """'!' when every identity the knowledge allows is playable, 'x' when every one is dead, else ''."""
    ids = [(c, r) for c in range(E.N_COLORS) if cm >> c & 1 for r in range(1, E.N_RANKS + 1) if rm >> (r - 1) & 1]
    if ids and all(fw[c] == r - 1 for c, r in ids):
        return "!"
    if ids and all(dead(c, r, fw, dcount) for c, r in ids):
        return "x"
    return ""


def knowledge(cm: int, rm: int) -> str:
    cs = "".join(E.COLORS[c] for c in range(E.N_COLORS) if cm >> c & 1)
    rs = "".join(str(r) for r in range(1, E.N_RANKS + 1) if rm >> (r - 1) & 1)
    return f"[{cs}|{rs}]"


def dcounts(discard) -> dict:
    return Counter((c.c, c.r) for c in discard)


# ------------------------------------------------------------------------------------------------- legal codes
def legal_codes(v: View) -> list:
    """(code, touched slots or None) in the engine's order -- from the view alone: plays and discards need only the
    own hand's size, hints only the other hands."""
    n = len(v.own)
    out = [(f"P{i}", None) for i in range(1, n + 1)]
    if v.info < E.MAX_TOKENS:
        out += [(f"D{i}", None) for i in range(1, n + 1)]
    if v.info > 0:
        for k, hand in enumerate(v.others, start=1):
            for val in E.hint_values(hand):
                out.append((f"H{k}{val}", tuple(i + 1 for i, hc in enumerate(hand) if E._matches(hc, val))))
    return out


# ------------------------------------------------------------------------------------------------- text
def _who(seat: int, obs_seat: int) -> str:
    k = (seat - obs_seat) % E.PLAYERS
    return "you" if k == 0 else f"P+{k}"


def describe(ev: E.Event, obs_seat: int) -> str:
    who = _who(ev.seat, obs_seat)
    if ev.kind == "P":
        return f"T{ev.turn} {who}: played {ev.card} from slot {ev.slot}, " + ("success" if ev.ok else "failed (life lost)")
    if ev.kind == "D":
        return f"T{ev.turn} {who}: discarded {ev.card} from slot {ev.slot}"
    sl = ",".join(str(i) for i in ev.touched)
    return f"T{ev.turn} {who}: hinted {_who(ev.target, obs_seat)} {ev.val} -> slot{'s' if len(ev.touched) > 1 else ''} {sl}"


def render(v: View) -> str:
    """The user message of one turn. Turns are numbered from 1 (seat 0 makes turn 1). In the final round (deck 0) K is
    the number of turns left including this one (turns to play before this move's decrement)."""
    if v.end is not None:
        raise E.EngineError("no observation of a finished game")
    dc = dcounts(v.discard)
    fin = f" Final round: {v.ttp} turns left including this one." if v.deck == 0 else ""
    lines = [f"Turn {v.turn}. Deck {v.deck}.{fin} Hint tokens {v.info}/{E.MAX_TOKENS}. Lives {v.lives}/{E.MAX_LIVES}. "
             f"Score {sum(v.fw)}.",
             "Fireworks: " + " ".join(f"{E.COLORS[c]}{v.fw[c]}" for c in range(E.N_COLORS)),
             "Discards: " + (" ".join(str(c) for c in v.discard) if v.discard else "none")]
    lines.append("You: " + " ".join(f"{i}:{'~' if t else ''}{knowledge(cm, rm)}{own_mark(cm, rm, v.fw, dc)}"
                                    for i, (cm, rm, t) in enumerate(v.own, start=1)))
    for k, hand in enumerate(v.others, start=1):
        lines.append(f"P+{k}: " + " ".join(
            f"{i}:{hc.card}{'~' if hc.touched else ''}{knowledge(hc.cm, hc.rm)}{marks(hc.card, v.fw, dc)}"
            for i, hc in enumerate(hand, start=1)))
    lines.append("Last actions:")
    lines += [" " + describe(ev, v.seat) for ev in v.hist] if v.hist else [" none yet"]
    lines.append("Legal actions:")
    lc = legal_codes(v)
    lines.append(" play: " + " ".join(c for c, _ in lc if c[0] == "P"))
    ds = [c for c, _ in lc if c[0] == "D"]
    if ds:
        lines.append(" discard: " + " ".join(ds))
    for k in range(1, E.PLAYERS):
        hs = [f"{c}({','.join(str(i) for i in t)})" for c, t in lc if c[0] == "H" and c[1] == str(k)]
        if hs:
            lines.append(f" hint P+{k}: " + " ".join(hs))
    lines.append("Your action:")
    return "\n".join(lines)


def observe(s: E.State, seat: Optional[int] = None) -> str:
    """The text the seat to act (or `seat`) sees."""
    return render(project(s, s.cur if seat is None else seat))


def messages(obs_text: str) -> list:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs_text}]


def prompt_ids(tok, obs_text: str) -> list:
    """The prompt token ids of one turn: the model directory's chat template with the thinking block switched off and
    the generation prompt added. `tok` is a transformers tokenizer (or a test double with apply_chat_template). The
    driver sends exactly these ids and the row builder recomputes them: one function for both."""
    out = tok.apply_chat_template(messages(obs_text), add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return [int(x) for x in out]


def ids_sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()
