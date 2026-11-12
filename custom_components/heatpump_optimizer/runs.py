"""Compressor starts and runs: the plan's start count, and the commit of a
relaxed plan to runs the pump can run (docs/design/2026-11-12-honest-starts-and-runs).

A start is an off-to-on edge of the compressor, whatever level it starts to;
a level change inside a run is none. ``start_count`` counts them on the combined
draw in a reading the solver can descend: ``r[k] = min(1, q[k] / floor)`` is
the share of step k the compressor runs, and each step starts the share of it
that follows the previous step's off share. On a plan whose steps are all off
or at the floor or above, that is the integer number of off-to-on edges; a
trickle under the floor costs ``r * (1 - r)`` per step and is never free.

``commit`` turns a relaxed plan -- one that books levels under the floor as
duty-cycle averages -- into an on/off mask of runs and pauses the pump can
run, at the relaxed plan's energy: a step lifted to the floor is paid for by
a step left out, never raised on its own.

Kept free of Home Assistant imports so it can be unit-tested directly.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from functools import partial
from typing import Any, NamedTuple

import numpy as np

from .batchmath import row_sums
from .thermal_model import MIN_RUNNING_DRAW_KW

#: The shortest run a plan may hold, and the shortest stop; the cut-off ends
#: no run and causes no stop shorter (``early_cutoff``). The plan's own step is
#: 15 minutes, so on that grid every run and pause of one step keeps both.
MIN_ON = timedelta(minutes=10)
MIN_OFF = timedelta(minutes=10)
#: ``commit``'s energy bands, in floor steps either side of the relaxed plan's
#: cumulative energy: tight tracking, a few hours' shift, and "the same energy
#: at any time". Constants from the design spike, not measured thresholds.
BANDS = (2, 6, 24)
#: The balance grid: cells per floor step of energy.
QUANTA_PER_STEP = 4
#: ``commit_store``'s temperature grid, kelvin per cell.
STORE_GRID_K = 0.25
#: What a step pays for leaving the relaxed plan's own on/off reading: a tie
#: break only, so a mask that saves nothing keeps the plan's runs.
TIE_BREAK = 1e-9


def floor_kw(floor: float) -> float:
    """The running floor a start is counted against; a configured 0 still runs
    at ``MIN_RUNNING_DRAW_KW``."""
    return max(float(floor), MIN_RUNNING_DRAW_KW)


def steps_at_least(span: timedelta, dt_hours: float) -> int:
    """Whole plan steps covering ``span``, at least one."""
    return max(1, int(np.ceil(span.total_seconds() / 3600.0 / dt_hours - 1e-9)))


def _share(power: np.ndarray, floor: float) -> np.ndarray:
    return np.clip(power / floor, 0.0, 1.0)


def _prior_share(share: np.ndarray, first: float) -> np.ndarray:
    """Each step's predecessor share, ``first`` before step 0."""
    out = np.empty_like(share)
    out[:, 0] = first
    out[:, 1:] = share[:, :-1]
    return out


def start_count(
    total: Any, floor: float, running: bool = False, dhw: Any = None,
    to_space: float = 0.0, to_dhw: float = 0.0,
) -> float:
    """The start count of one plan; the one-row case of ``start_count_batch``."""
    return float(start_count_batch(
        np.asarray(total, dtype=float)[None, :], floor, running, dhw, to_space, to_dhw,
    )[0])


def start_count_batch(
    total: np.ndarray, floor: float, running: bool = False, dhw: Any = None,
    to_space: float = 0.0, to_dhw: float = 0.0,
) -> np.ndarray:
    """``start_count`` for a [B, n] batch of combined draws, one entry per row.

    ``running``: the compressor runs at solve time, so a run continuing from
    it is no start, and a stop and a restart is one. ``dhw`` is the hot-water
    share of each row's draw; with ``to_space`` and ``to_dhw``, the learned
    probability that a duty change into that circuit stops the compressor,
    each duty change inside a run adds its direction's probability. A step
    with both is hot water first, so it holds one change into space. Both 0,
    the count is the run count alone. Elementwise to ``row_sums``, so each
    row is the scalar count bit for bit.
    """
    rows = np.asarray(total, dtype=float)
    f = floor_kw(floor)
    share = _share(rows, f)
    count: np.ndarray = row_sums(share * (1.0 - _prior_share(share, 1.0 if running else 0.0)))
    if dhw is None or (to_space <= 0.0 and to_dhw <= 0.0):
        return count
    hot = np.broadcast_to(np.asarray(dhw, dtype=float), rows.shape)
    s, d = _share(rows - hot, f), _share(hot, f)
    s0, d0 = _prior_share(s, 0.0), _prior_share(d, 0.0)
    into_space = d * s + d0 * (1.0 - s0) * s * (1.0 - d)
    return count + to_space * row_sums(into_space) + to_dhw * row_sums(s0 * d)


def realizable(total: Any, floor: float) -> bool:
    """Every step off or at the floor or above: the plan runs as it stands."""
    q = np.asarray(total, dtype=float)
    return bool(np.all((q <= 0.0) | (q >= floor_kw(floor) - 1e-9)))


def commit(
    q: Any, prices: Any, floor: float, caps: Any, start_cost: Any, band: int,
    *, dt: float, running: bool = False, forced: Any = None,
    lengths: tuple[int, int] = (1, 1),
) -> tuple[np.ndarray, bool]:
    """The on/off mask that runs the relaxed plan ``q`` cheapest, and whether
    it tracked within its band.

    A dynamic program over the steps with the state (on or off, run or pause
    length up to ``lengths``, the shortest run and pause in steps, energy
    balance). The balance
    is the committed energy less ``q``'s up to that step, on a grid of
    ``QUANTA_PER_STEP`` cells per floor step, held within ``band`` floor steps
    and within half a floor step at the end. An on step runs at ``max(q,
    floor)`` under ``caps`` and costs its price; an off-to-on edge costs
    ``start_cost`` (a scalar, or one per step). ``forced`` is the pin
    encoding (NaN free, 0 off, 1 on); a cap under the floor forces a step off.
    A forced step may cut a run or a pause short. Where no mask stays inside
    the band, the one nearest it is returned with ``False``: a best effort,
    never a raise.
    """
    q = np.asarray(q, dtype=float)
    n = q.size
    min_on, min_off = lengths
    f = floor_kw(floor)
    cap = np.broadcast_to(np.asarray(caps, dtype=float), (n,))
    want = np.full(n, np.nan) if forced is None else np.array(forced, dtype=float)[:n]
    want = np.where(np.isnan(want) & (cap < f - 1e-9), 0.0, want)
    level = np.minimum(np.maximum(q, f), cap)
    on_cost = np.asarray(prices, dtype=float)[:n] * level * dt
    start = np.broadcast_to(np.asarray(start_cost, dtype=float), (n,))
    reading = q > 0.0
    quantum = f * dt / QUANTA_PER_STEP
    half = QUANTA_PER_STEP * int(band)
    cost, bal = _origin(lengths, 2 * half + 1, running, half, 0.0)
    trail = []
    exact = True

    def place(b: np.ndarray, clip: bool) -> tuple[np.ndarray, np.ndarray]:
        j = np.rint(b / quantum).astype(np.int64) + half
        if clip:
            j = np.clip(j, 0, 2 * half)
        return j, (j >= 0) & (j <= 2 * half)

    for k in range(n):
        moves = _moves(min_on, min_off, want[k])
        on_inc, off_inc = (level[k] - q[k]) * dt, -q[k] * dt
        steps = (
            (lambda b, i=on_inc: b + i, on_cost[k] + TIE_BREAK * (not reading[k])),
            (lambda b, i=off_inc: b + i, TIE_BREAK * reading[k]),
        )
        nxt = _advance(cost, bal, moves, steps, float(start[k]), partial(place, clip=False), min_on)
        if not np.isfinite(nxt[0]).any():
            exact = False
            nxt = _advance(cost, bal, moves, steps, float(start[k]), partial(place, clip=True), min_on)
        cost, bal = nxt[0], nxt[1]
        trail.append(nxt[2])
    live = np.isfinite(cost)
    inside = live & (np.abs(bal) <= f * dt / 2.0 + 1e-12)
    keys: tuple[np.ndarray, ...] = (cost[inside],)
    if not inside.any():
        exact, inside = False, live
        keys = (cost[inside], np.abs(bal[inside]))
    order = np.lexsort(keys)
    u, j = (int(a[order[0]]) for a in np.nonzero(inside))
    return _unwind(trail, u, j, min_on), exact


class Store(NamedTuple):
    """The planner's linear store over the horizon (``dhw_planner``'s
    ``_plan_dhw_min_cost``): ``T[k+1] = decay * T[k] + drift[k]``, plus up to
    ``heat(k, T)`` kelvin on a charging step, and the bounds ``floors[k]``
    and ``ceilings[k]`` on ``T[k+1]``. A charge stops at the ceiling, as the
    pump's own controller ends it there."""

    initial: float
    decay: float
    drift: np.ndarray
    heat: Callable[[int, np.ndarray], np.ndarray]
    floors: np.ndarray
    ceilings: np.ndarray


def commit_store(
    store: Store, on_cost: np.ndarray, start_cost: Any,
    *, running: bool = False, forced: Any = None, lengths: tuple[int, int] = (1, 1),
) -> np.ndarray | None:
    """The cheapest charge mask that keeps ``store`` inside its bounds at
    every step, or ``None`` where none does; a charge that reaches the
    ceiling runs part of its step (``charge_shares``) and ends its run.

    ``commit``'s program with the store temperature, on a ``STORE_GRID_K``
    grid, in place of the energy balance: an on step costs ``on_cost`` for
    the share it runs and adds its heat, an off-to-on edge costs
    ``start_cost`` (scalar or per step), ``forced`` and ``lengths`` as there.
    Each cell keeps the exact temperature of the cheapest path into it, so
    rounding does not accrue.
    """
    n = np.size(on_cost)
    lo = min(store.initial, float(np.min(store.floors)))
    width = int(np.ceil((max(store.initial, float(np.max(store.ceilings))) - lo) / STORE_GRID_K)) + 1
    want = np.full(n, np.nan) if forced is None else np.array(forced, dtype=float)[:n]
    start = np.broadcast_to(np.asarray(start_cost, dtype=float), (n,))
    cost, temp = _origin(lengths, width, running, int(round((store.initial - lo) / STORE_GRID_K)), store.initial)
    trail = []
    for k in range(n):
        def place(t: np.ndarray, k: int = k) -> tuple[np.ndarray, np.ndarray]:
            j = np.rint((t - lo) / STORE_GRID_K).astype(np.int64)
            return j, (t >= store.floors[k]) & (t <= store.ceilings[k]) & (j >= 0) & (j < width)

        idle = (lambda t, k=k: store.decay * t + store.drift[k], 0.0)
        steps = ((lambda t, k=k: idle[0](t) + store.heat(k, t), float(on_cost[k])), idle)
        whole = _advance(cost, temp, _moves(*lengths, want[k]), steps, float(start[k]), place, lengths[0])
        topped = _advance(cost, temp, [] if want[k] == 0.0 else _tops(lengths), (
            steps[0], (lambda t, k=k: _charged(store, k, t)[0],
                       lambda t, k=k: float(on_cost[k]) * _charged(store, k, t)[1]),
        ), float(start[k]), place, lengths[0])
        top = topped[0] < whole[0]
        cost, temp = np.where(top, topped[0], whole[0]), np.where(top, topped[1], whole[1])
        if not np.isfinite(cost).any():
            return None
        trail.append(np.concatenate([np.where(top, topped[2], whole[2]), top[None]]))
    u, j = (int(a) for a in np.unravel_index(np.argmin(cost), cost.shape))
    return _unwind(trail, u, j, lengths[0])


def charge_shares(store: Store, mask: np.ndarray) -> np.ndarray:
    """The share of each step ``mask`` charges, walked on ``store``: 1 for a
    whole step, less where the charge reaches the ceiling, 0 off."""
    share, t = np.zeros(len(mask)), np.array([store.initial])
    for k, on in enumerate(mask):
        t, share[k] = (_charged(store, k, t)[0], float(_charged(store, k, t)[1][0])) if on else (
            store.decay * t + store.drift[k], 0.0)
    return share


def _tops(lengths: tuple[int, int]) -> list[tuple[int, int, bool]]:
    """The moves of a charge that reaches the ceiling: from any run, or with a
    start from any pause, into a pause, since the charge ends there."""
    min_on, states = lengths[0], sum(lengths)
    return [(i, min_on, i >= min_on) for i in range(states)]


def _charged(store: Store, k: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The store after a charging step ``k`` from ``t``, and the share of the
    step the charge ran before it reached the ceiling."""
    idle = store.decay * t + store.drift[k]
    heat = store.heat(k, t)
    end = np.minimum(idle + heat, np.maximum(idle, store.ceilings[k]))
    return end, np.clip((end - idle) / np.maximum(heat, 1e-12), 0.0, 1.0)


def _origin(
    lengths: tuple[int, int], width: int, running: bool, cell: int, value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The program's state before step 0: cost 0 in ``cell``, a long run
    or pause, as ``running`` says."""
    cost = np.full((sum(lengths), width), np.inf)
    val = np.zeros_like(cost)
    row = lengths[0] - 1 if running else sum(lengths) - 1
    cost[row, cell], val[row, cell] = 0.0, value
    return cost, val


def _unwind(trail: list[np.ndarray], u: int, j: int, min_on: int) -> np.ndarray:
    """The mask the back pointers in ``trail`` lead to from state ``(u, j)``;
    a third layer marks a step that charged into a pause."""
    mask = np.zeros(len(trail), dtype=bool)
    for k in range(len(trail) - 1, -1, -1):
        mask[k] = u < min_on or (len(trail[k]) > 2 and bool(trail[k][2, u, j]))
        u, j = (int(v) for v in trail[k][:2, u, j])
    return mask


def candidates(
    q: Any, prices: Any, floor: float, caps: Any, start_cost: Any,
    *, dt: float, running: bool = False, forced: Any = None,
) -> tuple[list[np.ndarray], bool]:
    """``commit``'s distinct masks over ``BANDS``, tightest first, at the
    ``MIN_ON`` and ``MIN_OFF`` step counts of ``dt``, and whether they track.
    Where a band holds no mask, the tightest best effort alone, with False."""
    masks: list[np.ndarray] = []
    lengths = (steps_at_least(MIN_ON, dt), steps_at_least(MIN_OFF, dt))
    for band in BANDS:
        mask, exact = commit(
            q, prices, floor, caps, start_cost, band, dt=dt, running=running,
            forced=forced, lengths=lengths,
        )
        if not exact:
            return masks or [mask], bool(masks)
        if not any(np.array_equal(mask, m) for m in masks):
            masks.append(mask)
    return masks, True


def _moves(min_on: int, min_off: int, want: float) -> list[tuple[int, int, bool]]:
    """``(source, destination, starts)`` state moves one step allows.

    States ``0 .. min_on - 1`` are a run of that many steps plus one (the last
    one long enough to stop), then the pauses alike. A forced step may leave
    any run or pause; otherwise only the long-enough one switches.
    """
    last_on, states = min_on - 1, min_on + min_off
    out: list[tuple[int, int, bool]] = []
    if not want == 0.0:
        out += [(i, min(i + 1, last_on), False) for i in range(min_on)]
        out += [(i, 0, True) for i in range(min_on, states) if not np.isnan(want) or i == states - 1]
    if not want == 1.0:
        out += [(i, min(i + 1, states - 1), False) for i in range(min_on, states)]
        out += [(i, min_on, False) for i in range(min_on) if not np.isnan(want) or i == last_on]
    return out


def _advance(
    cost: np.ndarray, val: np.ndarray, moves: list[tuple[int, int, bool]],
    steps: tuple[tuple[Callable[[np.ndarray], np.ndarray], Any], ...], start: float,
    place: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], min_on: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One step of the program: the cheapest arrival per cell, and where it
    came from. ``steps`` is ``(move, cost)`` on and off, ``move`` taking a
    cell's value to the next and ``cost`` a float or a function of the value; ``place`` gives each new value its cell and
    whether it is inside the bounds."""
    width = cost.shape[1]
    new_cost = np.full_like(cost, np.inf)
    new_bal = np.zeros_like(val)
    back = np.zeros((2,) + cost.shape, dtype=np.int32)
    cells = np.arange(width)
    for dest in sorted({m[1] for m in moves}):
        srcs = [(s, st) for s, d, st in moves if d == dest]
        move, extra = steps[0] if dest < min_on else steps[1]
        c = np.concatenate([
            cost[s] + (extra(val[s]) if callable(extra) else extra) + (start if st else 0.0)
            for s, st in srcs
        ])
        b = np.concatenate([move(val[s]) for s, _ in srcs])
        origin = np.concatenate([np.full(width, s) for s, _ in srcs])
        came = np.tile(cells, len(srcs))
        j, inside = place(b)
        ok = np.isfinite(c) & inside
        if not ok.any():
            continue
        c, b, origin, came, j = c[ok], b[ok], origin[ok], came[ok], j[ok]
        order = np.lexsort((c, j))
        first = order[np.r_[True, j[order][1:] != j[order][:-1]]]
        new_cost[dest, j[first]] = c[first]
        new_bal[dest, j[first]] = b[first]
        back[0, dest, j[first]] = origin[first]
        back[1, dest, j[first]] = came[first]
    return new_cost, new_bal, back
