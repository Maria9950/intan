# -*- coding: utf-8 -*-
import itertools
import numpy as np

EPS = 1e-12
def interval_contains(matrix, center, radius, eps=EPS):
    lower = center - radius
    upper = center + radius

    return np.all(
        (matrix >= lower - eps) &
        (matrix <= upper + eps)
    )

def matrix_rank(matrix, tol=1e-10):
    return np.linalg.matrix_rank(matrix, tol=tol)

def is_singular(matrix, tol=1e-10):
    return abs(np.linalg.det(matrix)) <= tol

def print_matrix(matrix, name):
    print(f"\n{name}:")
    print(np.array2string(
        matrix,
        precision=9,
        suppress_small=False
    ))

def analyze_a1_regression():

    print("=" * 70)
    print("A1 — ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 70)

    center = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    direction = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])

    print_matrix(center, "mid(A1)")
    print_matrix(direction, "Матрица радиусов")

    m_min = np.min(center[:, 0])
    m_max = np.max(center[:, 0])

    minimum_delta = (m_max - m_min) / 2.0

    z = (m_min + m_max) / 2.0

    print("\nПоиск минимального delta:")
    print(f"m_min (первый столбец) = {m_min:.6f}")
    print(f"m_max (первый столбец) = {m_max:.6f}")
    print(f"Разброс = {m_max - m_min:.6f}")
    print(f"Минимальное delta = {minimum_delta:.6f}")
    print(f"Значение z = {z:.6f}")

    a1_prime = np.column_stack([
        np.full(3, z),
        np.ones(3)
    ])

    radius = minimum_delta * direction

    print_matrix(a1_prime, "Построенная матрица A1'")

    # Проверки
    belongs = interval_contains(a1_prime, center, radius)
    rank = matrix_rank(a1_prime)

    print("\nПроверка результата:")
    print(f"A1' принадлежит A1: {belongs}")
    print(f"rank(A1') = {rank}")
    print(f"Требуется rank(A1') < 2: {rank < 2}")

    return minimum_delta, a1_prime

def analyze_a1_tomography():

    print("\n" + "=" * 70)
    print("A1 — МАЛОРАКУРСНАЯ ТОМОГРАФИЯ")
    print("=" * 70)

    center = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    direction = np.ones((3, 2))

    print_matrix(center, "mid(A1)")
    print_matrix(direction, "Матрица радиусов")

    m_min = np.min(center[:, 0])
    m_max = np.max(center[:, 0])
    c = center[0, 1]

    spread = m_max - m_min

    minimum_delta = spread / (m_max + m_min + 2.0 * c)

    z = (m_min + minimum_delta) / (c - minimum_delta)

    print("\nПоиск минимального delta:")
    print(f"m_min = {m_min:.6f}")
    print(f"m_max = {m_max:.6f}")
    print(f"c (центр 2-го столбца) = {c:.6f}")
    print(f"Разброс = {spread:.6f}")
    print(f"Минимальное delta = {minimum_delta:.12f}")
    print(f"Точно: delta = 1/27 = {1.0/27:.12f}")
    print(f"Коэффициент пропорциональности z = {z:.12f}")

    b = np.array([
        c - minimum_delta,
        c,
        c + minimum_delta
    ])

    a = z * b

    a1_prime = np.column_stack([a, b])
    radius = minimum_delta * direction

    print_matrix(a1_prime, "Построенная матрица A1'")

    belongs = interval_contains(a1_prime, center, radius)
    rank = matrix_rank(a1_prime)

    print("\nПроверка результата:")
    print(f"A1' принадлежит A1: {belongs}")
    print(f"rank(A1') = {rank}")
    print(f"Требуется rank(A1') < 2: {rank < 2}")

    print("\nПроверка линейной зависимости:")
    print(f"max|A1'[:,0] - z*A1'[:,1]| = "
          f"{np.max(np.abs(a - z * b)):.3e}")

    return minimum_delta, a1_prime
def find_singular_matrix_a2(center):

    inverse = np.linalg.inv(center)
    n = center.shape[0]

    best_value = -np.inf
    best_raw = 0.0
    best_y = None
    best_z = None

    for y in itertools.product([-1.0, 1.0], repeat=n):
        y = np.array(y)

        for z in itertools.product([-1.0, 1.0], repeat=n):
            z = np.array(z)

            raw_value = z @ inverse @ y
            value = abs(raw_value)

            if value > best_value:
                best_value = value
                best_raw = raw_value
                best_y = y.copy()
                best_z = z.copy()

    minimum_delta = 1.0 / best_value

    sign = np.sign(best_raw)  # +1 или -1

    perturbation = -minimum_delta * sign * np.outer(best_y, best_z)

    singular_matrix = center + perturbation

    return (
        minimum_delta,
        singular_matrix,
        perturbation,
        best_value,
        best_raw,
        best_y,
        best_z
    )


def analyze_a2():
    print("\n" + "=" * 70)
    print("A2 — КВАДРАТНАЯ МАТРИЦА")
    print("=" * 70)

    center = np.array([
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20]
    ])

    direction = np.ones((3, 3))

    print_matrix(center, "mid(A2)")
    print_matrix(direction, "Матрица радиусов")

    det_center = np.linalg.det(center)
    rank_center = matrix_rank(center)

    print("\nИсходная матрица:")
    print(f"det(mid(A2)) = {det_center:.12f}")
    print(f"rank(mid(A2)) = {rank_center}")

    (
        minimum_delta,
        a2_prime,
        perturbation,
        best_value,
        best_raw,
        y,
        z
    ) = find_singular_matrix_a2(center)

    print("\nПоиск минимального delta:")
    print(f"max |z^T A^(-1) y| = {best_value:.12f}")
    print(f"z^T A^(-1) y (с знаком) = {best_raw:.12f}")
    print(f"Минимальное delta = {minimum_delta:.12f}")

    print("\nЗнаковые векторы, определяющие критическое "
          "возмущение:")
    print(f"y = {y.astype(int)}")
    print(f"z = {z.astype(int)}")

    print_matrix(perturbation, "Критическое возмущение E")
    print_matrix(a2_prime, "Построенная матрица A2'")

    radius = minimum_delta * direction

    belongs = interval_contains(a2_prime, center, radius)
    determinant = np.linalg.det(a2_prime)
    rank = matrix_rank(a2_prime)

    print("\nПроверка результата:")
    print(f"A2' принадлежит A2: {belongs}")
    print(f"det(A2') = {determinant:.12e}")
    print(f"rank(A2') = {rank}")
    print(f"A2' вырожденная: {is_singular(a2_prime)}")

    return minimum_delta, a2_prime


def investigate_delta_a1_regression():

    center = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    delta_star = (
        np.max(center[:, 0]) -
        np.min(center[:, 0])
    ) / 2.0

    print("\n" + "-" * 70)
    print("ИССЛЕДОВАНИЕ A1 — РЕГРЕССИЯ")
    print("-" * 70)

    for delta in [
        0.0,
        delta_star / 2,
        delta_star - 0.001,
        delta_star,
        delta_star + 0.001,
        delta_star * 2
    ]:

        intersection_left = np.max(center[:, 0] - delta)
        intersection_right = np.min(center[:, 0] + delta)

        exists = intersection_left <= intersection_right + EPS

        print(
            f"delta = {delta:.6f} | "
            f"пересечение: "
            f"[{intersection_left:.6f}, "
            f"{intersection_right:.6f}] | "
            f"вырожденная матрица существует: {exists}"
        )

def investigate_delta_a2():
    center = np.array([
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20]
    ])

    direction = np.ones((3, 3))

    delta_star, _, _, _, _, _, _ = find_singular_matrix_a2(center)

    print("\n" + "-" * 70)
    print("ИССЛЕДОВАНИЕ A2")
    print("-" * 70)

    print(f"Критическое значение delta* = {delta_star:.12f}")

    delta_values = [
        0,
        delta_star / 2,
        delta_star - 0.001,
        delta_star,
        delta_star + 0.001,
        2 * delta_star
    ]

    print("\nЧисленное исследование различных значений delta:")

    for delta in delta_values:

        radius = delta * direction

        lower = center - radius
        upper = center + radius

        print("\n" + "-" * 70)
        print(f"delta = {delta:.6f}")

        print("Интервальная матрица A2:")

        for i in range(3):
            print(
                f"[{lower[i, 0]:.4f}; {upper[i, 0]:.4f}]  "
                f"[{lower[i, 1]:.4f}; {upper[i, 1]:.4f}]  "
                f"[{lower[i, 2]:.4f}; {upper[i, 2]:.4f}]"
            )


        if delta < delta_star - 1e-12:
            print("Сингулярная точечная матрица: НЕ существует")
        else:
            print("Сингулярная точечная матрица: существует")

    print("\n" + "-" * 70)
    print("ТЕОРЕТИЧЕСКИЙ ВЫВОД")
    print("-" * 70)

    print(
        "При delta < delta* интервальная матрица "
        "не содержит сингулярных точечных матриц."
    )

    print(
        "При delta = delta* появляется первая "
        "сингулярная точечная матрица."
    )

    print(
        "При delta > delta* интервальная матрица "
        "также содержит сингулярные точечные матрицы."
    )

def print_conclusion(
    delta_regression,
    delta_tomography,
    delta_square
):
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    print(
        f"\nA1 — линейная регрессия:"
        f"\n  delta* = {delta_regression:.12f}"
        f"\n  delta* = 0.075"
    )

    print(
        f"\nA1 — малоракурсная томография:"
        f"\n  delta* = {delta_tomography:.12f}"
        f"\n  delta* = 1/27 ≈ {1.0/27:.12f}"
    )

    print(
        f"\nA2 — квадратная система:"
        f"\n  delta* = {delta_square:.12f}"
    )

    print("\nИнтерпретация:")

    print(
        "\n1. В случае линейной регрессии неопределённость "
        "есть только в первом столбце. Для появления "
        "линейной зависимости столбцов первый столбец "
        "должен стать постоянным. Это требует delta* = 0.075 "
        "(половина разброса центров первого столбца)."
    )

    print(
        "\n2. В случае малоракурсной томографии неопределённы "
        "оба столбца A1. Поэтому можно одновременно "
        "изменять оба столбца и получить линейную "
        "зависимость уже при меньшем delta* = 1/27 ≈ 0.037. "
        "Двойка в знаменателе формулы (m_max + m_min + 2c) "
        "возникает потому, что центр второго столбца c = 1 "
        "входит в выражение дважды — по одному разу из "
        "каждой границы пропорции z."
    )

    print(
        "\n3. Для квадратной матрицы A2 исходная матрица "
        "невырождена, однако при критическом радиусе "
        "delta в интервальной матрице появляется "
        "сингулярная точечная матрица. Поиск ведётся "
        "перебором знаковых векторов y, z ∈ {-1,+1}^3 "
        "с максимизацией |z^T A^{-1} y|."
    )

if __name__ == "__main__":

    delta_regression, _ = analyze_a1_regression()

    delta_tomography, _ = analyze_a1_tomography()

    delta_square, _ = analyze_a2()

    investigate_delta_a1_regression()
    investigate_delta_a2()

    print_conclusion(
        delta_regression,
        delta_tomography,
        delta_square
    )