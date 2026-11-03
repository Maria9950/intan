from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

OUT = Path("lab3_figures")
OUT.mkdir(exist_ok=True)

SYSTEMS = {
    "A1": {
        "A_lo": np.array([
            [0.65, 0.70],
            [0.75, 0.74],
        ], dtype=float),
        "A_hi": np.array([
            [1.25, 1.30],
            [1.35, 1.26],
        ], dtype=float),
        "b_lo": np.array([0.96, 1.00], dtype=float),
        "b_hi": np.array([1.01, 1.05], dtype=float),
    },
    "A2": {
        "A_lo": np.array([
            [0.65, 0.70],
            [0.75, 0.74],
            [0.80, 0.79],
        ], dtype=float),
        "A_hi": np.array([
            [1.25, 1.30],
            [1.35, 1.26],
            [1.40, 1.31],
        ], dtype=float),
        "b_lo": np.array([0.96, 1.00, 1.02], dtype=float),
        "b_hi": np.array([1.01, 1.05, 1.07], dtype=float),
    },
    "A3": {
        "A_lo": np.array([
            [0.65, 0.70],
            [0.75, 0.74],
            [0.80, 0.79],
            [-0.30, 0.64],
        ], dtype=float),
        "A_hi": np.array([
            [1.25, 1.30],
            [1.35, 1.26],
            [1.40, 1.31],
            [0.30, 1.24],
        ], dtype=float),
        "b_lo": np.array([0.96, 1.00, 1.02, 0.63], dtype=float),
        "b_hi": np.array([1.01, 1.05, 1.07, 0.68], dtype=float),
    },
}


def mid_rad(lo, hi):
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def endpoints(mid, rad):
    return mid - rad, mid + rad


def row_values(x, A_lo, A_hi, b_lo, b_hi):
    A_mid, A_rad = mid_rad(A_lo, A_hi)
    b_mid, b_rad = mid_rad(b_lo, b_hi)
    x = np.asarray(x, dtype=float)
    return b_rad - A_rad @ np.abs(x) - np.abs(b_mid - A_mid @ x)


def tol_value(x, A_lo, A_hi, b_lo, b_hi):
    return np.min(row_values(x, A_lo, A_hi, b_lo, b_hi))


def maximize_tol(A_lo, A_hi, b_lo, b_hi):
    A_mid, A_rad = mid_rad(A_lo, A_hi)
    b_mid, b_rad = mid_rad(b_lo, b_hi)

    m, n = A_mid.shape
    total = n + n + m + 1

    x0 = 0
    u0 = n
    v0 = 2 * n
    t_idx = 2 * n + m

    A_ub = []
    b_ub = []

    for j in range(n):
        row = np.zeros(total)
        row[x0 + j] = 1.0
        row[u0 + j] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(total)
        row[x0 + j] = -1.0
        row[u0 + j] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    for i in range(m):
        row = np.zeros(total)
        row[x0:x0+n] = -A_mid[i]
        row[v0 + i] = -1.0
        A_ub.append(row)
        b_ub.append(-b_mid[i])

        row = np.zeros(total)
        row[x0:x0+n] = A_mid[i]
        row[v0 + i] = -1.0
        A_ub.append(row)
        b_ub.append(b_mid[i])

        row = np.zeros(total)
        row[u0:u0+n] = A_rad[i]
        row[v0 + i] = 1.0
        row[t_idx] = 1.0
        A_ub.append(row)
        b_ub.append(b_rad[i])

    c = np.zeros(total)
    c[t_idx] = -1.0

    bounds = (
        [(None, None)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * m
        + [(None, None)]
    )

    result = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    tau = result.x[:n]
    T = result.x[t_idx]
    F = row_values(tau, A_lo, A_hi, b_lo, b_hi)
    active = np.where(np.isclose(F, T, atol=1e-8))[0] + 1

    return tau, T, F, active


def corrected_data(A_lo, A_hi, b_lo, b_hi, kind, alpha=0.5):
    A_mid, A_rad = mid_rad(A_lo, A_hi)
    b_mid, b_rad = mid_rad(b_lo, b_hi)

    tau, T, _, _ = maximize_tol(A_lo, A_hi, b_lo, b_hi)

    delta = 0.25 * np.min(b_rad)
    D = -T + delta
    norm1 = np.sum(np.abs(tau))

    if kind == "b":
        A_rad_new = A_rad.copy()
        b_rad_new = b_rad + D
        params = {"delta": delta, "D": D, "mu_b": D}

    elif kind == "A":
        e_max = np.min(A_rad)
        e_required = D / norm1
        e_A = min(e_required, e_max * (1.0 - 1e-6))
        mu_A = D - e_A * norm1

        A_rad_new = A_rad - e_A
        b_rad_new = b_rad + mu_A
        params = {
            "delta": delta,
            "D": D,
            "e_required": e_required,
            "e_max": e_max,
            "e_A": e_A,
            "mu_A": mu_A,
        }

    elif kind == "Ab":
        e_max = np.min(A_rad)
        D_A = alpha * D
        e_Ab = min(D_A / norm1, e_max * (1.0 - 1e-6))
        mu_Ab = D - e_Ab * norm1

        A_rad_new = A_rad - e_Ab
        b_rad_new = b_rad + mu_Ab
        params = {
            "delta": delta,
            "D": D,
            "alpha": alpha,
            "e_max": e_max,
            "e_Ab": e_Ab,
            "mu_Ab": mu_Ab,
        }

    else:
        raise ValueError("kind must be 'b', 'A' or 'Ab'")

    A_lo_new, A_hi_new = endpoints(A_mid, A_rad_new)
    b_lo_new, b_hi_new = endpoints(b_mid, b_rad_new)

    tau_new, T_new, F_new, active_new = maximize_tol(
        A_lo_new, A_hi_new, b_lo_new, b_hi_new
    )

    return {
        "A_lo": A_lo_new,
        "A_hi": A_hi_new,
        "b_lo": b_lo_new,
        "b_hi": b_hi_new,
        "A_rad": A_rad_new,
        "b_rad": b_rad_new,
        "tau": tau_new,
        "T": T_new,
        "F": F_new,
        "active": active_new,
        "params": params,
    }


def tol_grid(A_lo, A_hi, b_lo, b_hi, x1, x2):
    A_mid, A_rad = mid_rad(A_lo, A_hi)
    b_mid, b_rad = mid_rad(b_lo, b_hi)

    X1, X2 = np.meshgrid(x1, x2)
    values = []

    for i in range(A_mid.shape[0]):
        mid_Ax = A_mid[i, 0] * X1 + A_mid[i, 1] * X2
        rad_Ax = A_rad[i, 0] * np.abs(X1) + A_rad[i, 1] * np.abs(X2)
        Fi = b_rad[i] - rad_Ax - np.abs(b_mid[i] - mid_Ax)
        values.append(Fi)

    return X1, X2, np.minimum.reduce(values)


def plot_surface(name, label, A_lo, A_hi, b_lo, b_hi, tau, T):
    span_x = max(0.8, 0.35 * abs(tau[0]) + 0.5)
    span_y = max(0.8, 0.35 * abs(tau[1]) + 0.5)

    x1 = np.linspace(tau[0] - span_x, tau[0] + span_x, 180)
    x2 = np.linspace(tau[1] - span_y, tau[1] + span_y, 180)
    X1, X2, Z = tol_grid(A_lo, A_hi, b_lo, b_hi, x1, x2)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Z, linewidth=0, antialiased=True)
    ax.scatter([tau[0]], [tau[1]], [T], s=45)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Tol")
    ax.set_title(f"{name}: Tol(x), {label}\nmax Tol = {T:.6f}")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{label}_tol.png", dpi=200)
    plt.close(fig)


def plot_tolerance_set(name, label, A_lo, A_hi, b_lo, b_hi, tau, T):
    span_x = max(0.35, 0.18 * abs(tau[0]) + 0.25)
    span_y = max(0.35, 0.18 * abs(tau[1]) + 0.25)

    x1 = np.linspace(tau[0] - span_x, tau[0] + span_x, 600)
    x2 = np.linspace(tau[1] - span_y, tau[1] + span_y, 600)
    X1, X2, Z = tol_grid(A_lo, A_hi, b_lo, b_hi, x1, x2)

    fig, ax = plt.subplots(figsize=(8, 6))

    if np.max(Z) > 0:
        ax.contourf(X1, X2, Z, levels=[0, np.max(Z) + 1e-12], alpha=0.35)

    if np.min(Z) <= 0 <= np.max(Z):
        ax.contour(X1, X2, Z, levels=[0], linewidths=1.5)

    ax.scatter([tau[0]], [tau[1]], marker="*", s=120)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{name}: допусковое множество, {label}\nmax Tol = {T:.6f}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{label}_set.png", dpi=200)
    plt.close(fig)


def print_matrix(name, M):
    print(name)
    print(np.array2string(np.asarray(M), precision=8, suppress_small=True))


def analyze_system(name, data):
    A_lo = data["A_lo"]
    A_hi = data["A_hi"]
    b_lo = data["b_lo"]
    b_hi = data["b_hi"]

    A_mid, A_rad = mid_rad(A_lo, A_hi)
    b_mid, b_rad = mid_rad(b_lo, b_hi)

    tau, T, F, active = maximize_tol(A_lo, A_hi, b_lo, b_hi)

    print("\n" + "=" * 70)
    print(name)
    print_matrix("mid(A):", A_mid)
    print_matrix("rad(A):", A_rad)
    print_matrix("mid(b):", b_mid)
    print_matrix("rad(b):", b_rad)
    print(f"T = {T:.9f}")
    print(f"tau = ({tau[0]:.9f}, {tau[1]:.9f})")
    print_matrix("F_i(tau):", F)
    print("Активные строки:", active.tolist())
    print(f"Дефицит = {abs(T):.9f}")

    plot_surface(name, "original", A_lo, A_hi, b_lo, b_hi, tau, T)

    for kind in ("b", "A", "Ab"):
        result = corrected_data(A_lo, A_hi, b_lo, b_hi, kind)

        print("\n", kind, "коррекция")
        for key, value in result["params"].items():
            print(f"{key} = {value:.9f}")

        print_matrix("rad(A'):", result["A_rad"])
        print_matrix("rad(b'):", result["b_rad"])
        print(f"T' = {result['T']:.9f}")
        print(f"tau' = ({result['tau'][0]:.9f}, {result['tau'][1]:.9f})")
        print("Активные строки:", result["active"].tolist())

        plot_surface(
            name,
            kind,
            result["A_lo"],
            result["A_hi"],
            result["b_lo"],
            result["b_hi"],
            result["tau"],
            result["T"],
        )
        plot_tolerance_set(
            name,
            kind,
            result["A_lo"],
            result["A_hi"],
            result["b_lo"],
            result["b_hi"],
            result["tau"],
            result["T"],
        )


if __name__ == "__main__":
    for system_name, system_data in SYSTEMS.items():
        analyze_system(system_name, system_data)

    print("\nГрафики сохранены в папке:", OUT.resolve())
