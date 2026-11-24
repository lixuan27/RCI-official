"""Hanabi rules engine: the standard four-player game of the Hanabi Challenge (Bard et al., 2020) with the Hanabi
Learning Environment's default parameters (5 colours x ranks 1-5, card counts 3/2/2/2/1, 8 information tokens, 3 lives,
4 cards a hand, seat 0 moves first). An independent implementation of those rules; no code of the environment is used.

Pure Python, deterministic; a state is an immutable NamedTuple and `apply` returns a new state.
  hints   name another player and one colour or rank present in that hand; cost a token; every card learns "is" or
          "is not" that colour / rank
  discard legal only below 8 tokens; returns a token
  play    always legal: a playable card joins its firework, any other card is discarded and costs a life; completing
          a firework (its 5) returns a token unless 8 are held. Both draw while the deck is not empty
  end     no lives left, all fireworks complete (25), or after the last draw every player (the drawer too) has taken
          one more turn (turns_to_play, decremented before a move whenever the deck is already empty)
  score   the challenge score is 0 when the lives ran out, else the cards in the fireworks; score_pl keeps the cards
          played whatever ended the game
  hands   keep their order: a card leaving a hand moves later cards up one slot, a drawn card goes last
The deal is a uniform permutation of the 50 cards from a SHA-256 stream: one seed gives one deck on every platform.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple, Optional, Tuple

COLORS = "RYGWB"
RANKS = "12345"
N_COLORS = 5
N_RANKS = 5
COPIES = (0, 3, 2, 2, 2, 1)          # COPIES[r]: cards of one color with rank r
PLAYERS = 4
HAND_SIZE = 4                         # fewer than 4 players would hold 5
MAX_TOKENS = 8
MAX_LIVES = 3
MAX_SCORE = 25
ALL = 0b11111                         # every color / every rank still possible
N_CARDS = 50


class EngineError(ValueError):
    pass


class Card(NamedTuple):
    c: int      # color index 0..4 (R Y G W B)
    r: int      # rank 1..5

    def __str__(self) -> str:
        return f"{COLORS[self.c]}{self.r}"


class HandCard(NamedTuple):
    card: Card
    cm: int      # colours still possible, bit c (positive and negative hints)
    rm: int      # ranks still possible, bit r-1
    ch: bool     # a color hint has touched it
    rh: bool     # a rank hint has touched it

    @property
    def touched(self) -> bool:
        return self.ch or self.rh


class Move(NamedTuple):
    kind: str            # "P" play, "D" discard, "H" hint
    slot: int = 0        # P / D: 1-based slot of the acting player's hand
    k: int = 0           # H: the target's offset after the actor, 1..3
    val: str = ""        # H: a color letter of COLORS or a rank digit of RANKS


class Event(NamedTuple):
    """One move as every player saw it (all of it is public: a played or discarded card is shown to everyone)."""
    turn: int
    seat: int
    kind: str
    slot: int = 0
    card: Optional[Card] = None
    ok: bool = False             # P: the card was playable
    target: int = -1             # H: absolute seat of the hinted player
    val: str = ""
    touched: Tuple[int, ...] = ()


class State(NamedTuple):
    order: Tuple[Card, ...]                 # the whole deck order (hidden from every player)
    nxt: int                                # index of the next card to draw
    hands: Tuple[Tuple[HandCard, ...], ...]
    fw: Tuple[int, ...]                     # top rank of each color's firework (0 = empty)
    discard: Tuple[Card, ...]               # in time order
    info: int
    lives: int
    cur: int                                # the seat to act
    ttp: int                                # turns to play once the deck is empty
    turn: int                               # number of the next move, from 1 (seat 0 makes move 1)
    hist: Tuple[Event, ...]
    end: Optional[str]                      # None while running; "lives" | "perfect" | "deck"

    @property
    def deck(self) -> int:
        return len(self.order) - self.nxt


# ------------------------------------------------------------------------------------------------- the deal
def canonical_deck() -> Tuple[Card, ...]:
    """R Y G W B, each [1, 1, 1, 2, 2, 3, 3, 4, 4, 5]."""
    return tuple(Card(c, r) for c in range(N_COLORS) for r in range(1, N_RANKS + 1) for _ in range(COPIES[r]))


def _uniform(seed: int, i: int, n: int) -> int:
    """An unbiased integer in [0, n) from SHA-256 by rejection sampling."""
    lim = (2 ** 64 // n) * n
    k = 0
    while True:
        v = int.from_bytes(hashlib.sha256(f"hanabi/deal/{seed}/{i}/{k}".encode()).digest()[:8], "big")
        if v < lim:
            return v % n
        k += 1


def deck_order(seed: int) -> Tuple[Card, ...]:
    """Fisher-Yates over the canonical deck, i = 49 .. 1, j = U_i(i + 1)."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise EngineError(f"a deal seed is a non-negative int, not {seed!r}")
    d = list(canonical_deck())
    for i in range(len(d) - 1, 0, -1):
        j = _uniform(seed, i, i + 1)
        d[i], d[j] = d[j], d[i]
    return tuple(d)


def order_sha(order) -> str:
    return hashlib.sha256(",".join(str(c) for c in order).encode()).hexdigest()


def new_game_from_order(order) -> State:
    """Seat p's initial slot s is order[4p + s - 1]; draws then follow from order[16]."""
    order = tuple(Card(int(c[0]), int(c[1])) for c in order)
    if sorted(order) != sorted(canonical_deck()):
        raise EngineError("the order is not a permutation of the 50 cards")
    hands = tuple(tuple(HandCard(order[HAND_SIZE * p + s], ALL, ALL, False, False) for s in range(HAND_SIZE))
                  for p in range(PLAYERS))
    return State(order, PLAYERS * HAND_SIZE, hands, (0,) * N_COLORS, (), MAX_TOKENS, MAX_LIVES, 0, PLAYERS, 1, (), None)


def new_game(seed: int) -> State:
    return new_game_from_order(deck_order(seed))


# ------------------------------------------------------------------------------------------------- moves and codes
def code(m: Move) -> str:
    if m.kind in ("P", "D"):
        return f"{m.kind}{m.slot}"
    return f"H{m.k}{m.val}"


def all_codes() -> Tuple[str, ...]:
    """Every code the grammar can produce in a 4-player game: P1-P4, D1-D4, H<1-3><RYGWB12345> (38)."""
    return tuple([f"P{s}" for s in range(1, 5)] + [f"D{s}" for s in range(1, 5)]
                 + [f"H{k}{v}" for k in range(1, 4) for v in COLORS + RANKS])


def _matches(hc: HandCard, val: str) -> bool:
    if val in COLORS:
        return hc.card.c == COLORS.index(val)
    return hc.card.r == int(val)


def hint_values(hand) -> Tuple[str, ...]:
    """The hint values present in a hand, colors R Y G W B first, then ranks 1..5 (the legal-move order)."""
    cs = {hc.card.c for hc in hand}
    rs = {hc.card.r for hc in hand}
    return tuple([COLORS[c] for c in range(N_COLORS) if c in cs] + [str(r) for r in range(1, N_RANKS + 1) if r in rs])


def legal(s: State) -> list:
    """The legal moves in code order: P1.., D1.. (fewer than 8 tokens), then hints to P+1, P+2, P+3 (a token held)."""
    if s.end is not None:
        return []
    n = len(s.hands[s.cur])
    out = [Move("P", i) for i in range(1, n + 1)]
    if s.info < MAX_TOKENS:
        out += [Move("D", i) for i in range(1, n + 1)]
    if s.info > 0:
        for k in range(1, PLAYERS):
            out += [Move("H", 0, k, v) for v in hint_values(s.hands[(s.cur + k) % PLAYERS])]
    return out


def legal_codes(s: State) -> list:
    return [code(m) for m in legal(s)]


def parse(text: str, s: State) -> Move:
    """A code back to the move; a code that is not legal in `s` is refused."""
    for m in legal(s):
        if code(m) == text:
            return m
    raise EngineError(f"{text!r} is not a legal code at turn {s.turn}")


def playable(card: Card, fw) -> bool:
    return fw[card.c] == card.r - 1


def _draw(s_order, nxt, hand):
    if nxt < len(s_order):
        return hand + (HandCard(s_order[nxt], ALL, ALL, False, False),), nxt + 1
    return hand, nxt


def apply(s: State, m: Move) -> State:
    """The state after move m of the seat to act (refused when the game is over or m is not legal)."""
    if s.end is not None:
        raise EngineError("the game is over")
    if m not in legal(s):
        raise EngineError(f"{code(m)} is not legal at turn {s.turn}")
    ttp = s.ttp - 1 if s.deck == 0 else s.ttp          # before the move, only when the deck is already empty
    hands = list(s.hands)
    me = hands[s.cur]
    fw, discard, info, lives, nxt = list(s.fw), s.discard, s.info, s.lives, s.nxt
    if m.kind == "H":
        tgt = (s.cur + m.k) % PLAYERS
        new, touched = [], []
        for i, hc in enumerate(hands[tgt]):
            hit = _matches(hc, m.val)
            if m.val in COLORS:
                bit = 1 << COLORS.index(m.val)
                hc = hc._replace(cm=bit, ch=True) if hit else hc._replace(cm=hc.cm & ~bit)
            else:
                bit = 1 << (int(m.val) - 1)
                hc = hc._replace(rm=bit, rh=True) if hit else hc._replace(rm=hc.rm & ~bit)
            if hit:
                touched.append(i + 1)
            new.append(hc)
        hands[tgt] = tuple(new)
        info -= 1
        ev = Event(s.turn, s.cur, "H", target=tgt, val=m.val, touched=tuple(touched))
    else:
        card = me[m.slot - 1].card
        rest = me[:m.slot - 1] + me[m.slot:]
        ok = False
        if m.kind == "P":
            if playable(card, fw):
                ok = True
                fw[card.c] += 1
                if fw[card.c] == N_RANKS and info < MAX_TOKENS:
                    info += 1
            else:
                discard = discard + (card,)
                lives -= 1
        else:
            discard = discard + (card,)
            info += 1
        rest, nxt = _draw(s.order, nxt, rest)
        hands[s.cur] = rest
        ev = Event(s.turn, s.cur, m.kind, slot=m.slot, card=card, ok=ok)
    end = None
    if lives <= 0:
        end = "lives"
    elif sum(fw) >= MAX_SCORE:
        end = "perfect"
    elif ttp <= 0:
        end = "deck"
    return State(s.order, nxt, tuple(hands), tuple(fw), discard, info, lives, (s.cur + 1) % PLAYERS, ttp,
                 s.turn + 1, s.hist + (ev,), end)


def apply_code(s: State, text: str) -> State:
    return apply(s, parse(text, s))


# ------------------------------------------------------------------------------------------------- scores
def score_pl(s: State) -> int:
    return sum(s.fw)


def score_ch(s: State) -> int:
    """The challenge score: 0 when the lives ran out."""
    return 0 if s.lives <= 0 else sum(s.fw)


def replay(seed_or_order, codes) -> list:
    """Every state of a game from its deal and its codes: states[t] is the state before the (t+1)-th move."""
    s = new_game(seed_or_order) if isinstance(seed_or_order, int) else new_game_from_order(seed_or_order)
    out = [s]
    for c in codes:
        s = apply_code(s, c)
        out.append(s)
    return out


def multiset_ok(s: State) -> bool:
    """Conservation: hands + deck + fireworks + discard pile = the 50 cards."""
    cards = [hc.card for h in s.hands for hc in h] + list(s.order[s.nxt:]) + list(s.discard)
    cards += [Card(c, r) for c in range(N_COLORS) for r in range(1, s.fw[c] + 1)]
    return sorted(cards) == sorted(canonical_deck())
