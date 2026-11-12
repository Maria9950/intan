from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

N = 14
RAD = 1.0 / 2 ** N
EPS = 1e-7
HEADER_SIZE = 256
FRAME_SIZE = 16400
FRAME_HEADER = 16
POINTS = 1024
CHANNELS = 8


def load_bin(path):
    raw = Path(path).read_bytes()
    frame_count = int(np.frombuffer(raw[2:4], dtype="<u2")[0])
    data = raw[HEADER_SIZE:HEADER_SIZE + frame_count * FRAME_SIZE]
    frames = np.frombuffer(data, dtype=np.uint8).reshape(frame_count, FRAME_SIZE)
    codes = frames[:, FRAME_HEADER:].copy().view("<u2").reshape(frame_count, POINTS, CHANNELS)
    return codes.astype(np.float64).ravel() / 16384.0 - 0.5


def make_intervals(values):
    values = np.asarray(values, dtype=np.float64)
    return values - RAD, values + RAD


def meet(x, y):
    return max(x[0], y[0]), min(x[1], y[1])


def join(x, y):
    return min(x[0], y[0]), max(x[1], y[1])


def jaccard(x, y):
    m = meet(x, y)
    j = join(x, y)
    return (m[1] - m[0]) / (j[1] - j[0])


def shift(x, a):
    return x[0] + a, x[1] + a


def scale(x, t):
    if t >= 0:
        return t * x[0], t * x[1]
    return t * x[1], t * x[0]


def elementary_intervals(lo, hi):
    points = np.unique(np.concatenate((lo, hi)))
    z_lo = points[:-1]
    z_hi = points[1:]
    lo_sorted = np.sort(lo)
    hi_sorted = np.sort(hi)
    mu = (
        np.searchsorted(lo_sorted, z_lo, side="right")
        - np.searchsorted(hi_sorted, z_lo, side="right")
    )
    return z_lo, z_hi, mu


def interval_mode(lo, hi):
    z_lo, z_hi, mu = elementary_intervals(lo, hi)
    mu_max = int(mu.max())
    idx = np.flatnonzero(mu == mu_max)
    return (float(z_lo[idx[0]]), float(z_hi[idx[-1]])), mu_max, (z_lo, z_hi, mu)


def median_kreinovich(lo, hi):
    return float(np.median(lo)), float(np.median(hi))


def median_prolubnikov(lo, hi):
    z_lo, z_hi, mu = elementary_intervals(lo, hi)
    mask = mu > 0
    z_lo = z_lo[mask]
    z_hi = z_hi[mask]
    mu = mu[mask].astype(np.int64)
    cumulative = np.cumsum(mu)
    total = int(cumulative[-1])
    half = total / 2.0
    k = int(np.searchsorted(cumulative, half, side="left"))
    if cumulative[k] == half and k + 1 < len(mu):
        return (
            float((z_lo[k] + z_lo[k + 1]) / 2.0),
            float((z_hi[k] + z_hi[k + 1]) / 2.0),
        )
    return float(z_lo[k]), float(z_hi[k])


def filter_3sigma(values):
    med = float(np.median(values))
    sigma = float(np.std(values))
    mask = np.abs(values - med) <= 3.0 * sigma
    return values[mask], int((~mask).sum())


def sample_extrema(values):
    lo, hi = make_intervals(values)
    return float(lo.min()), float(lo.max()), float(hi.min()), float(hi.max())


def transform_extrema(extrema, parameter, s):
    lo_min, lo_max, hi_min, hi_max = extrema
    if parameter == "a":
        return lo_min + s, lo_max + s, hi_min + s, hi_max + s
    if s >= 0:
        return s * lo_min, s * lo_max, s * hi_min, s * hi_max
    return s * hi_max, s * hi_min, s * lo_max, s * lo_min


def sample_function(parameter, x, y):
    x_ext = sample_extrema(x)
    y_ext = sample_extrema(y)

    def F(s):
        tx = transform_extrema(x_ext, parameter, s)
        meet_lo = max(tx[1], y_ext[1])
        meet_hi = min(tx[2], y_ext[2])
        join_lo = min(tx[0], y_ext[0])
        join_hi = max(tx[3], y_ext[3])
        return (meet_hi - meet_lo) / (join_hi - join_lo)

    return F


def interval_function(parameter, x, y):
    operation = shift if parameter == "a" else scale

    def F(s):
        return jaccard(operation(x, s), y)

    return F


def golden_max(F, left, right):
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - phi * (right - left)
    d = left + phi * (right - left)
    fc = F(c)
    fd = F(d)
    while right - left > EPS:
        if fc >= fd:
            right, d, fd = d, c, fc
            c = right - phi * (right - left)
            fc = F(c)
        else:
            left, c, fc = c, d, fd
            d = left + phi * (right - left)
            fd = F(d)
    s = (left + right) / 2.0
    return float(s), float(F(s))


def argmax(F, center, half_width):
    left = center - half_width
    right = center + half_width
    grid = np.linspace(left, right, 2001)
    values = np.array([F(float(s)) for s in grid])
    max_value = float(values.max())
    idx = np.flatnonzero(np.isclose(values, max_value, rtol=0.0, atol=1e-12))
    if len(idx) > 1:
        s = float((grid[idx[0]] + grid[idx[-1]]) / 2.0)
        return s, float(F(s))
    k = int(idx[0])
    for _ in range(2):
        step = grid[1] - grid[0]
        local_left = grid[max(0, k - 2)]
        local_right = grid[min(len(grid) - 1, k + 2)]
        grid = np.linspace(local_left, local_right, 2001)
        values = np.array([F(float(s)) for s in grid])
        k = int(np.argmax(values))
    step = grid[1] - grid[0]
    return golden_max(F, grid[k] - 2 * step, grid[k] + 2 * step)


def bisect_level(F, inside, outside, level):
    while abs(outside - inside) > EPS:
        middle = (inside + outside) / 2.0
        if F(middle) >= level:
            inside = middle
        else:
            outside = middle
    return float(inside)


def level_interval(F, s_hat, level, step):
    left_step = step
    left_out = s_hat - left_step
    while F(left_out) >= level:
        left_step *= 1.6
        left_out = s_hat - left_step
    left = bisect_level(F, s_hat, left_out, level)

    right_step = step
    right_out = s_hat + right_step
    while F(right_out) >= level:
        right_step *= 1.6
        right_out = s_hat + right_step
    right = bisect_level(F, s_hat, right_out, level)

    return left, right


def estimate(F, parameter, case, center, candidate=None):
    half_width = 0.15 if parameter == "a" else 0.75
    s_hat, f_max = argmax(F, center, half_width)
    if candidate is not None and F(candidate) > f_max:
        s_hat = float(candidate)
        f_max = float(F(candidate))
    level = (f_max - 1.0) / 2.0 if case == "B1" else 0.0
    step = 0.01 if case == "B1" and parameter == "a" else 0.1 if case == "B1" else 5e-5
    interval = level_interval(F, s_hat, level, step)
    return s_hat, f_max, level, interval


def plot_histograms(x, y, mode_x, mode_y, medk_x, medk_y, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, mode, medk, name, hist_color in (
        (axes[0], x, mode_x, medk_x, "X", "tab:blue"),
        (axes[1], y, mode_y, medk_y, "Y", "lightgreen"),
    ):
        med = float(np.median(values))
        bins = np.arange(med - 0.04, med + 0.04 + 4 * RAD, 4 * RAD)
        ax.hist(values, bins=bins, alpha=0.75, color=hist_color)
        ax.axvspan(mode[0], mode[1], alpha=0.45, color="red", label="mode")
        ax.axvline(np.mean(medk), color="black", linestyle="--", linewidth=1.2, label="mid medK")
        ax.set_xlim(med - 0.04, med + 0.04)
        ax.set_xlabel("V, В")
        ax.set_ylabel("Количество")
        ax.set_title(f"Выборка {name}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hist_XY.png", dpi=160)
    plt.close(fig)

def plot_frequencies(elem_x, elem_y, mode_x, mode_y, mu_x, mu_y, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, elem, mode, mu_max, name, line_color in (
        (axes[0], elem_x, mode_x, mu_x, "X", "tab:blue"),
        (axes[1], elem_y, mode_y, mu_y, "Y", "lightgreen"),
    ):
        z_lo, z_hi, mu = elem
        mask = (z_lo >= mode[0] - 0.02) & (z_hi <= mode[1] + 0.02)
        ax.step(z_lo[mask], mu[mask], where="post", color=line_color)
        ax.axvspan(mode[0], mode[1], alpha=0.45, color="red", label=f"mode, μ = {mu_max}")
        ax.set_xlabel("z, В")
        ax.set_ylabel("μ")
        ax.set_title(f"Частота μ(z) для {name}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mu_XY.png", dpi=160)
    plt.close(fig)

def plot_function(F, result, case, parameter, out):
    s_hat, f_max, level, interval = result
    width = interval[1] - interval[0]
    half = 1.3 * max(s_hat - interval[0], interval[1] - s_hat) if case == "B1" else max(1.5 * width, 2e-4)
    s = np.linspace(s_hat - half, s_hat + half, 1601)
    f = np.array([F(float(v)) for v in s])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, f)
    ax.scatter([s_hat], [f_max], s=35, label=f"{parameter}max = {s_hat:.7f}")
    ax.axhline(level, linestyle="--", linewidth=1)
    ax.axvspan(interval[0], interval[1], alpha=0.18, label=f"[{interval[0]:.7f}, {interval[1]:.7f}]")
    ax.set_xlabel(parameter)
    ax.set_ylabel("F")
    ax.set_title(f"{case}: F({parameter})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"F_{case}_{parameter}.png", dpi=160)
    plt.close(fig)


def plot_raw_b1(F, parameter, center, out):
    half = 0.75 if parameter == "a" else 2.0
    grid = np.linspace(center - half, center + half, 1801)
    values = np.array([F(float(v)) for v in grid])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, values)
    ax.set_xlabel(parameter)
    ax.set_ylabel("F")
    ax.set_title(f"B.1 без удаления выбросов: F({parameter})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"F_B1_{parameter}_raw.png", dpi=160)
    plt.close(fig)

def describe(values):
    return float(values.min()), float(values.max()), float(values.mean()), float(np.median(values)), float(values.std())


def main(file_x, file_y, out_dir="lab4_results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = load_bin(file_x)
    y = load_bin(file_y)
    x_lo, x_hi = make_intervals(x)
    y_lo, y_hi = make_intervals(y)

    mode_x, mu_x, elem_x = interval_mode(x_lo, x_hi)
    mode_y, mu_y, elem_y = interval_mode(y_lo, y_hi)
    medk_x = median_kreinovich(x_lo, x_hi)
    medk_y = median_kreinovich(y_lo, y_hi)
    medp_x = median_prolubnikov(x_lo, x_hi)
    medp_y = median_prolubnikov(y_lo, y_hi)

    raw_b1_a = sample_function("a", x, y)
    raw_b1_t = sample_function("t", x, y)

    x_b1, removed_x = filter_3sigma(x)
    y_b1, removed_y = filter_3sigma(y)

    a_center = float(np.median(y) - np.median(x))
    t_center = float(np.median(y) / np.median(x))

    cases = {
        "B1": {
            "a": sample_function("a", x_b1, y_b1),
            "t": sample_function("t", x_b1, y_b1),
        },
        "B2": {
            "a": interval_function("a", mode_x, mode_y),
            "t": interval_function("t", mode_x, mode_y),
        },
        "B3": {
            "a": interval_function("a", medk_x, medk_y),
            "t": interval_function("t", medk_x, medk_y),
        },
        "B4": {
            "a": interval_function("a", medp_x, medp_y),
            "t": interval_function("t", medp_x, medp_y),
        },
    }

    stats = {
        "B2": (mode_x, mode_y),
        "B3": (medk_x, medk_y),
        "B4": (medp_x, medp_y),
    }

    results = {}
    for case in ("B1", "B2", "B3", "B4"):
        results[case] = {}
        for parameter in ("a", "t"):
            candidate = None
            if case != "B1":
                xi, yi = stats[case]
                if parameter == "a":
                    candidate = (yi[0] + yi[1] - xi[0] - xi[1]) / 2.0
                elif xi[1] < 0 < yi[0]:
                    candidate = (yi[0] / xi[1] + yi[1] / xi[0]) / 2.0
            center = a_center if parameter == "a" else t_center
            result = estimate(cases[case][parameter], parameter, case, center, candidate)
            results[case][parameter] = result
            plot_function(cases[case][parameter], result, case, parameter, out)

    plot_histograms(x, y, mode_x, mode_y, medk_x, medk_y, out)
    plot_frequencies(elem_x, elem_y, mode_x, mode_y, mu_x, mu_y, out)
    plot_raw_b1(raw_b1_a, "a", a_center, out)
    plot_raw_b1(raw_b1_t, "t", t_center, out)

    print(f"n = {x.size}, rad = {RAD:.10f} В")
    print("X: min, max, mean, median, std =", describe(x))
    print("Y: min, max, mean, median, std =", describe(y))
    print(f"mode X = {mode_x}, mu = {mu_x}")
    print(f"mode Y = {mode_y}, mu = {mu_y}")
    print(f"medK X = {medk_x}; medK Y = {medk_y}")
    print(f"medP X = {medp_x}; medP Y = {medp_y}")
    print(f"B1: удалено X = {removed_x}, Y = {removed_y}")

    for case in ("B1", "B2", "B3", "B4"):
        for parameter in ("a", "t"):
            s_hat, f_max, level, interval = results[case][parameter]
            print(
                f"{case} {parameter}: s_hat={s_hat:.7f}, Fmax={f_max:.6f}, "
                f"level={level:.6f}, interval=[{interval[0]:.7f}, {interval[1]:.7f}]"
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Запуск: python lab4.py <X.bin> <Y.bin> [папка_результатов]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "lab4_results")
