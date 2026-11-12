"""Spike for docs/design/2026-11-12-honest-starts-and-runs: relax, commit, dispatch.

Throwaway evidence for the design note's "Spike" table; it never merges.
Run from the repository root:

    PYTHONPATH=tests/hastub python3 dev/audit/harnesses/honest_starts_spike.py

Per profile it prints today's plan (modulating with the swing term, then
on_off), the relaxed plan with no start term, and each committed candidate:
the mask from ``commit`` below, its levels solved by ``optimize`` with the
mask handed in as manual space pins. ``starts`` is a start at every running
step whose predecessor did not run the whole step.
"""
import sys, time
sys.path.insert(0, 'tests'); sys.path.insert(0, 'custom_components')
import numpy as np
import golden
from profiles import solve_inputs
from heatpump_optimizer import optimizer as O
from heatpump_optimizer.thermal_model import MIN_RUNNING_DRAW_KW

DT = 0.25

def honest_starts(q, f, on0=False):
    """A start at every running step whose predecessor did not run the whole step."""
    n, prev_full = 0, on0
    for x in q:
        if x > MIN_RUNNING_DRAW_KW and not prev_full:
            n += 1
        prev_full = x >= f - 1e-9
    return n

def commit(q, price, f, cap, sigma, beta_q, on0=False):
    """On/off mask tracking the relaxed plan's cumulative energy within +-beta_q quanta."""
    n = len(q); delta = f * DT / 4.0; K = int(beta_q * 4)
    lvl = np.clip(np.maximum(q, f), f, cap)
    INF = float('inf')
    # cost[u][b], real balance bal[u][b], back pointers
    cost = [[INF] * (2 * K + 1) for _ in range(2)]
    bal = [[0.0] * (2 * K + 1) for _ in range(2)]
    cost[1 if on0 else 0][K] = 0.0
    back = []
    for k in range(n):
        ncost = [[INF] * (2 * K + 1) for _ in range(2)]
        nbal = [[0.0] * (2 * K + 1) for _ in range(2)]
        nback = [[None] * (2 * K + 1) for _ in range(2)]
        for u in (0, 1):
            for b in range(2 * K + 1):
                c = cost[u][b]
                if c == INF:
                    continue
                for v in (0, 1):
                    nb = bal[u][b] + ((lvl[k] if v else 0.0) - q[k]) * DT
                    j = int(round(nb / delta)) + K
                    if j < 0 or j > 2 * K:
                        continue
                    nc = c + (price[k] * lvl[k] * DT if v else 0.0) + (sigma if v and not u else 0.0)
                    if nc < ncost[v][j]:
                        ncost[v][j], nbal[v][j], nback[v][j] = nc, nb, (u, b)
        cost, bal = ncost, nbal
        back.append(nback)
    best = None
    for u in (0, 1):
        for b in range(2 * K + 1):
            if cost[u][b] < INF and abs(bal[u][b]) <= f * DT / 2 + 1e-9:
                if best is None or cost[u][b] < best[0]:
                    best = (cost[u][b], u, b)
    if best is None:  # relax the terminal parity
        best = min((cost[u][b] + 1e3 * abs(bal[u][b]), u, b) for u in (0, 1) for b in range(2 * K + 1) if cost[u][b] < INF)
    _, u, b = best
    mask = np.zeros(n)
    for k in range(n - 1, -1, -1):
        mask[k] = u
        u, b = back[k][u][b]
    return mask

def solve(built, **kw):
    opt = built['optimizer']
    t = time.perf_counter()
    res = opt.optimize(inputs=solve_inputs(
        initial_state=built['state'], prices=built['prices'], outdoor_temps=built['outdoor'],
        wind_speeds=built['wind'], precipitation=built['rain'], solar_radiation=built['solar'],
        start_time=golden.START, **kw))
    return res, time.perf_counter() - t

def row(tag, res, wall, f, prices, sigma, lo):
    q = np.array(res.power_schedule)
    sub = int(np.sum((q > MIN_RUNNING_DRAW_KW) & (q < f - 1e-9)))
    room = np.array(res.room_temp_trajectory)
    n = honest_starts(q, f)
    print(f"{tag:34s} starts {n:3d} subfloor {sub:3d} kWh {q.sum()*DT:6.2f} energy {float(np.sum(prices*q)*DT):7.2f} "
          f"min_room {room.min():5.2f} (floor {lo}) obj {res.objective_value:8.3f} wall {wall:5.2f}s")
    return n

def run(label, sigma, f, p_max, **spec):
    print(f"== {label}: sigma {sigma}, floor {f}, max {p_max}")
    base = dict(dhw=False, param_overrides={'min_electrical_power': f, 'max_electrical_power': p_max}, **spec)
    lo = golden.make(**base)['optimizer'].config.min_temp
    # today, modulating, legacy swing term
    b = golden.make(**base, opt_overrides={'cycling_cost': sigma}); r, w = solve(b)
    row('today modulating (swing term)', r, w, f, b['prices'], sigma, lo)
    # today, on_off projection
    b = golden.make(**base, opt_overrides={'cycling_cost': sigma, 'clamp_planned_levels': True}); r, w = solve(b)
    row('today on_off (raise to floor)', r, w, f, b['prices'], sigma, lo)
    # relaxed, no start term
    b = golden.make(**base); relaxed, w0 = solve(b)
    n0 = row('relaxed (no start term)', relaxed, w0, f, b['prices'], sigma, lo)
    q = np.array(relaxed.power_schedule)
    best = None
    t_all = 0.0
    seen = set()
    for beta in (2, 6, 24):
        t = time.perf_counter(); mask = commit(q, b['prices'], f, p_max, sigma, beta); t_dp = time.perf_counter() - t
        key = mask.tobytes()
        if key in seen:
            print(f"  beta {beta}: same mask"); continue
        seen.add(key)
        bb = golden.make(**base); res, w = solve(bb, space_pins=mask)
        t_all += w + t_dp
        n = row(f"committed beta={beta} (dp {t_dp*1e3:.0f} ms)", res, w, f, b['prices'], sigma, lo)
        total = res.objective_value + sigma * n
        print(f"    honest total {total:8.3f}   released pins {len(res.manual_released_space)}")
        if best is None or total < best[0]:
            best = (total, beta)
    print(f"  relaxed honest total {relaxed.objective_value + sigma * n0:8.3f}; best committed {best[0]:8.3f} (beta {best[1]}); added wall {t_all:.2f}s on {w0:.2f}s")

run('shoulder weather, flat prices', 2.5, 1.1, 3.3, price_profile='flat', weather_profile='shoulder')
run('shoulder weather, shoulder prices', 2.5, 1.1, 3.3, price_profile='shoulder', weather_profile='shoulder')
run('mild winter, typical prices', 2.5, 1.1, 3.3, price_profile='winter_typical', weather_profile='winter_mild')

