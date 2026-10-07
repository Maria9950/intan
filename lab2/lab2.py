# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

class Interval:
    def __init__(self, low, high=None):
        if high is None:
            high = low

        low = float(low)
        high = float(high)

        if low > high:
            low, high = high, low

        self.low = low
        self.high = high

    @property
    def mid(self):
        return (self.low + self.high) / 2

    @property
    def rad(self):
        return (self.high - self.low) / 2

    @property
    def wid(self):
        return self.high - self.low

    def __add__(self, other):
        if isinstance(other, (int, float, np.number)):
            return Interval(
                self.low + other,
                self.high + other
            )

        return Interval(
            self.low + other.low,
            self.high + other.high
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, np.number)):
            return Interval(
                self.low - other,
                self.high - other
            )

        return Interval(
            self.low - other.high,
            self.high - other.low
        )

    def __rsub__(self, other):
        if isinstance(other, (int, float, np.number)):
            return Interval(
                other - self.high,
                other - self.low
            )

        return Interval(
            other.low - self.high,
            other.high - self.low
        )

    def __neg__(self):
        return Interval(-self.high, -self.low)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.number)):
            a = self.low * other
            b = self.high * other

            return Interval(
                min(a, b),
                max(a, b)
            )

        products = [
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high
        ]

        return Interval(
            min(products),
            max(products)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError(
                "Поддерживаются только неотрицательные целые степени"
            )

        if power == 0:
            return Interval(1)

        # Нечётная степень монотонна
        if power % 2 == 1:
            return Interval(
                self.low ** power,
                self.high ** power
            )

        # Чётная степень
        if self.low <= 0 <= self.high:
            return Interval(
                0,
                max(abs(self.low), abs(self.high)) ** power
            )

        values = [
            self.low ** power,
            self.high ** power
        ]

        return Interval(
            min(values),
            max(values)
        )

    def sin(self):
        a = self.low
        b = self.high

        if b - a >= 2 * math.pi:
            return Interval(-1, 1)

        values = [
            math.sin(a),
            math.sin(b)
        ]

        # Экстремумы sin:
        # pi/2 + k*pi
        k_min = math.ceil(
            (a - math.pi / 2) / math.pi
        )

        k_max = math.floor(
            (b - math.pi / 2) / math.pi
        )

        for k in range(k_min, k_max + 1):
            x = math.pi / 2 + k * math.pi

            if a <= x <= b:
                values.append(math.sin(x))

        return Interval(
            min(values),
            max(values)
        )

    def cos(self):
        a = self.low
        b = self.high

        if b - a >= 2 * math.pi:
            return Interval(-1, 1)

        values = [
            math.cos(a),
            math.cos(b)
        ]

        # Экстремумы cos:
        # k*pi
        k_min = math.ceil(a / math.pi)
        k_max = math.floor(b / math.pi)

        for k in range(k_min, k_max + 1):
            x = k * math.pi

            if a <= x <= b:
                values.append(math.cos(x))

        return Interval(
            min(values),
            max(values)
        )

    def intersect(self, other):
        low = max(self.low, other.low)
        high = min(self.high, other.high)

        if low > high:
            return None

        return Interval(low, high)

    @staticmethod
    def hull(A, B):
        return Interval(
            min(A.low, B.low),
            max(A.high, B.high)
        )

    def __repr__(self):
        return (
            f"[{self.low:.6f}, "
            f"{self.high:.6f}]"
        )


def hausdorff_dist(A, B):
    """
    Расстояние Хаусдорфа между интервалами
    A=[a1,a2], B=[b1,b2].
    """

    return max(
        abs(A.low - B.low),
        abs(A.high - B.high)
    )


def cut(value, bounds=(-1.0, 1.0)):
    """
    Функция срезки для теоремы Бауманна.
    """

    return max(
        bounds[0],
        min(bounds[1], value)
    )


def baumann_p(derivative_interval):
    if derivative_interval.rad == 0:
        return 0.0

    return cut(
        derivative_interval.mid /
        derivative_interval.rad
    )


def bisection_root(
        func,
        a,
        b,
        tol=1e-13,
        max_iter=200
):
    """
    Метод бисекции для поиска корня.
    """

    fa = func(a)
    fb = func(b)

    if abs(fa) < tol:
        return a

    if abs(fb) < tol:
        return b

    if fa * fb > 0:
        raise ValueError(
            "На концах интервала нет смены знака"
        )

    left = a
    right = b

    for _ in range(max_iter):
        middle = (left + right) / 2
        fm = func(middle)

        if (
            abs(fm) < tol or
            right - left < tol
        ):
            return middle

        if fa * fm <= 0:
            right = middle
            fb = fm

        else:
            left = middle
            fa = fm

    return (left + right) / 2


def find_roots_on_interval(
        func,
        a,
        b,
        grid_size=20000
):
    """
    Численный поиск всех корней
    на заданном отрезке.
    """

    xs = np.linspace(
        a,
        b,
        grid_size + 1
    )

    ys = np.array([
        func(float(x))
        for x in xs
    ])

    roots = []

    for i in range(grid_size):

        x1 = float(xs[i])
        x2 = float(xs[i + 1])

        y1 = ys[i]
        y2 = ys[i + 1]

        if abs(y1) < 1e-10:
            roots.append(x1)

        if y1 * y2 < 0:

            root = bisection_root(
                func,
                x1,
                x2
            )

            roots.append(root)

    if abs(ys[-1]) < 1e-10:
        roots.append(float(xs[-1]))

    # Удаляем совпадающие корни

    roots_unique = []

    for root in sorted(roots):

        if (
            len(roots_unique) == 0 or
            abs(root - roots_unique[-1])
            > 1e-7
        ):
            roots_unique.append(root)

    return roots_unique


def print_separator(title):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_result(
        name,
        interval_result,
        exact
):

    print(
        f"{name:<38}"
        f" F(X)={str(interval_result):<25}"
        f" rad={interval_result.rad:10.6f}"
        f" dist={hausdorff_dist(interval_result, exact):10.6f}"
    )



def solve_f1():

    print_separator(
        "1.1. f1(x) = x^3 - 3x^2 + 2"
    )

    X = Interval(0, 3)

    exact = Interval(-2, 2)

    def f(x):
        return (
            x ** 3
            - 3 * x ** 2
            + 2
        )

    def df(x):
        return (
            3 * x ** 2
            - 6 * x
        )


    print(f"X = {X}")

    print(
        "ran(f1,X) =",
        exact
    )

    print(
        f"mid={exact.mid:.6f}, "
        f"rad={exact.rad:.6f}, "
        f"wid={exact.wid:.6f}"
    )


    f_eir = (
        X ** 3
        - 3 * X ** 2
        + 2
    )


    # x^3 - 3x^2 + 2 =
    # x^2(x-3)+2

    f_horner = (
        X ** 2
        * (X - 3)
        + 2
    )

    # Сдвиг:
    #
    # y = x - 1
    #
    # f = y^3 - 3y

    Y = X - 1

    f_shift = (
        Y ** 3
        - 3 * Y
    )

    print()
    print("Естественные интервальные расширения:")

    print_result(
        "ЕИР исходное",
        f_eir,
        exact
    )

    print_result(
        "Форма x^2(x-3)+2",
        f_horner,
        exact
    )

    print_result(
        "Сдвиг y=x-1",
        f_shift,
        exact
    )


    df_X = (
        3
        * X
        * (X - 2)
    )

    print()
    print(
        "f1'(X) =",
        df_X
    )

    centers = [
        0,
        X.mid,
        3
    ]

    mv_results = {}

    print()
    print("F_mv:")

    for c in centers:

        Fmv = (
            f(c)
            + df_X * (X - c)
        )

        mv_results[c] = Fmv

        print_result(
            f"c={c:.2f}",
            Fmv,
            exact
        )




    def slope_interval(c):

        return (
            X ** 2
            + (c - 3) * X
            + c ** 2
            - 3 * c
        )

    sl_results = {}

    print()
    print("F_sl:")

    for c in centers:

        slope = slope_interval(c)

        Fsl = (
            f(c)
            + slope * (X - c)
        )

        sl_results[c] = Fsl

        print(
            f"c={c:.2f}: "
            f"slope={slope}, "
            f"F_sl={Fsl}, "
            f"rad={Fsl.rad:.6f}, "
            f"dist="
            f"{hausdorff_dist(Fsl, exact):.6f}"
        )


    p = baumann_p(df_X)

    c_star = (
        X.mid
        - p * X.rad
    )

    c_star2 = (
        X.mid
        + p * X.rad
    )

    F_star = (
        f(c_star)
        + df_X
        * (X - c_star)
    )

    F_star2 = (
        f(c_star2)
        + df_X
        * (X - c_star2)
    )

    F_bic = (
        F_star
        .intersect(F_star2)
    )

    print()
    print("Форма Бауманна:")

    print(
        "mid f'(X) =",
        df_X.mid
    )

    print(
        "rad f'(X) =",
        df_X.rad
    )

    print(
        "p =",
        p
    )

    print(
        "c_* =",
        c_star
    )

    print(
        "c^* =",
        c_star2
    )

    print(
        "F_mv(c_*) =",
        F_star
    )

    print(
        "F_mv(c^*) =",
        F_star2
    )

    print_result(
        "F_bic",
        F_bic,
        exact
    )



    L = 9

    lipschitz_limit = (
        L * X.rad
    )

    print()
    print("Липшицевский анализ:")

    print(
        "L =",
        L
    )

    print(
        "rad X =",
        X.rad
    )

    print(
        "L * rad X =",
        lipschitz_limit
    )

    print(
        "rad ЕИР =",
        f_eir.rad
    )

    print(
        "rad Горнер =",
        f_horner.rad
    )

    print(
        "rad сдвиговой формы =",
        f_shift.rad
    )

    xx = np.linspace(
        X.low,
        X.high,
        1000
    )

    yy = f(xx)

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        xx,
        yy,
        linewidth=2,
        label=
        r"$f_1(x)=x^3-3x^2+2$"
    )

    plt.axhline(
        exact.low,
        linestyle="--",
        label="Точный минимум"
    )

    plt.axhline(
        exact.high,
        linestyle="--",
        label="Точный максимум"
    )

    plt.axhspan(
        f_eir.low,
        f_eir.high,
        alpha=0.10,
        label=f"ЕИР {f_eir}"
    )

    plt.axhspan(
        F_bic.low,
        F_bic.high,
        alpha=0.12,
        label=f"Бауманн {F_bic}"
    )

    plt.xlabel("x")
    plt.ylabel("f1(x)")

    plt.title(
        "Функция f1 и интервальные оценки"
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "fig_f1.png",
        dpi=180
    )

    plt.close()

    print()
    print("Сводные результаты f1:")

    print_result(
        "Точная область",
        exact,
        exact
    )

    print_result(
        "ЕИР",
        f_eir,
        exact
    )

    print_result(
        "Эквивалентная форма",
        f_horner,
        exact
    )

    print_result(
        "Сдвиг",
        f_shift,
        exact
    )

    print_result(
        "F_mv(mid X)",
        mv_results[X.mid],
        exact
    )

    print_result(
        "F_sl(mid X)",
        sl_results[X.mid],
        exact
    )

    print_result(
        "Бауманн",
        F_bic,
        exact
    )




def solve_f2():

    print_separator(
        "f2(x)=(x+1)^3-cos(x)"
    )

    X = Interval(-2, 1)

    def f(x):

        return (
            (x + 1) ** 3
            - np.cos(x)
        )

    def df(x):

        return (
            3 * (x + 1) ** 2
            + np.sin(x)
        )

    def d2f(x):

        return (
            6 * (x + 1)
            + np.cos(x)
        )


    critical_points = (
        find_roots_on_interval(
            lambda x:
            3 * (x + 1) ** 2
            + math.sin(x),

            X.low,
            X.high
        )
    )

    inflection_points = (
        find_roots_on_interval(
            lambda x:
            6 * (x + 1)
            + math.cos(x),

            X.low,
            X.high
        )
    )

    candidates = (
        [X.low, X.high]
        + critical_points
    )

    values = [
        (
            x,
            float(f(x))
        )
        for x
        in candidates
    ]

    x_min, f_min = min(
        values,
        key=lambda item:
        item[1]
    )

    x_max, f_max = max(
        values,
        key=lambda item:
        item[1]
    )

    exact = Interval(
        f_min,
        f_max
    )

    print(
        "Выбран X =",
        X
    )

    print()
    print(
        "Критические точки:"
    )

    for x in critical_points:

        print(
            f"x={x:.12f}, "
            f"f(x)={f(x):.12f}"
        )

    print()
    print(
        "Точки перегиба:"
    )

    for x in inflection_points:

        print(
            f"x={x:.12f}, "
            f"f(x)={f(x):.12f}"
        )

    print()
    print(
        "Кандидаты на глобальные "
        "экстремумы:"
    )

    for x, y in sorted(values):

        print(
            f"x={x:.12f}, "
            f"f(x)={y:.12f}"
        )

    print()
    print(
        "ran(f2,X) =",
        exact
    )

    print(
        f"Минимум: "
        f"x={x_min:.12f}"
    )

    print(
        f"Максимум: "
        f"x={x_max:.12f}"
    )

    print(
        f"mid={exact.mid:.6f}, "
        f"rad={exact.rad:.6f}, "
        f"wid={exact.wid:.6f}"
    )



    f_horner = (
        ((X + 3) * X + 3)
        * X
        + 1
        - X.cos()
    )

    print()
    print(
        "Естественные интервальные расширения:"
    )

    print_result(
        "ЕИР исходное",
        f_eir,
        exact
    )

    print_result(
        "Форма Горнера",
        f_horner,
        exact
    )


    df_X = (
        3 * ((X + 1) ** 2)
        + X.sin()
    )

    print()
    print(
        "f2'(X) =",
        df_X
    )

    centers = [
        X.low,
        X.mid,
        X.high
    ]

    mv_results = {}

    print()
    print("F_mv:")

    for c in centers:

        Fmv = (
            float(f(c))
            + df_X * (X - c)
        )

        mv_results[c] = Fmv

        print_result(
            f"c={c:.2f}",
            Fmv,
            exact
        )


    def slope_interval(c):

        U = X + 1

        polynomial_slope = (
            U ** 2
            + (c + 1) * U
            + (c + 1) ** 2
        )

        trig_slope = (
            X.sin()
        )

        return (
            polynomial_slope
            + trig_slope
        )

    sl_results = {}

    print()
    print("F_sl:")

    for c in centers:

        slope = (
            slope_interval(c)
        )

        Fsl = (
            float(f(c))
            + slope
            * (X - c)
        )

        sl_results[c] = Fsl

        print(
            f"c={c:.2f}: "
            f"slope={slope}, "
            f"F_sl={Fsl}, "
            f"rad={Fsl.rad:.6f}, "
            f"dist="
            f"{hausdorff_dist(Fsl, exact):.6f}"
        )


    p = baumann_p(
        df_X
    )

    c_star = (
        X.mid
        - p * X.rad
    )

    c_star2 = (
        X.mid
        + p * X.rad
    )

    F_star = (
        float(f(c_star))
        + df_X
        * (X - c_star)
    )

    F_star2 = (
        float(f(c_star2))
        + df_X
        * (X - c_star2)
    )

    F_bic = (
        F_star
        .intersect(F_star2)
    )

    print()
    print(
        "Форма Бауманна:"
    )

    print(
        "mid f'(X)=",
        df_X.mid
    )

    print(
        "rad f'(X)=",
        df_X.rad
    )

    print(
        "p=",
        p
    )

    print(
        "c_*=",
        c_star
    )

    print(
        "c^*=",
        c_star2
    )

    print(
        "F_mv(c_*)=",
        F_star
    )

    print(
        "F_mv(c^*)=",
        F_star2
    )

    print_result(
        "F_bic",
        F_bic,
        exact
    )


    lipschitz_candidates = (
        [X.low, X.high]
        + inflection_points
    )

    L = max(
        abs(float(df(x)))
        for x
        in lipschitz_candidates
    )

    lipschitz_limit = (
        L * X.rad
    )

    print()
    print(
        "Липшиц:"
    )

    print(
        "L =",
        L
    )

    print(
        "rad X =",
        X.rad
    )

    print(
        "L * rad X =",
        lipschitz_limit
    )

    print(
        "rad ЕИР =",
        f_eir.rad
    )

    print(
        "rad Горнера =",
        f_horner.rad
    )


    xx = np.linspace(
        X.low,
        X.high,
        1500
    )

    yy = f(xx)

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        xx,
        yy,
        linewidth=2,
        label=
        r"$f_2(x)=(x+1)^3-\cos x$"
    )

    plt.axhline(
        exact.low,
        linestyle="--",
        label=
        f"Точный min={exact.low:.4f}"
    )

    plt.axhline(
        exact.high,
        linestyle="--",
        label=
        f"Точный max={exact.high:.4f}"
    )

    plt.axhspan(
        f_eir.low,
        f_eir.high,
        alpha=0.12,
        label=f"ЕИР {f_eir}"
    )

    plt.scatter(
        critical_points,
        [
            f(x)
            for x
            in critical_points
        ],
        zorder=5,
        label="Критические точки"
    )

    plt.xlabel("x")
    plt.ylabel("f2(x)")

    plt.title(
        "f2 на X=[-2,1]"
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "fig_f2_variant12.png",
        dpi=180
    )

    plt.close()

    print()
    print(
        "Сводные результаты f2:"
    )

    print_result(
        "Точная область",
        exact,
        exact
    )

    print_result(
        "ЕИР",
        f_eir,
        exact
    )

    print_result(
        "Горнер",
        f_horner,
        exact
    )

    print_result(
        "F_mv(mid X)",
        mv_results[X.mid],
        exact
    )

    print_result(
        "F_sl(mid X)",
        sl_results[X.mid],
        exact
    )

    print_result(
        "Бауманн",
        F_bic,
        exact
    )



def solve_f3():

    print_separator(
        "f3(x,y)=x^2*y+y^2-x"
    )

    X = Interval(-1, 2)
    Y = Interval(-1, 2)

    def f(x, y):

        return (
            x ** 2 * y
            + y ** 2
            - x
        )



    exact = Interval(
        -5,
        10
    )

    print(
        f"Брус: {X} x {Y}"
    )

    print()
    print(
        "df/dx = 2xy - 1"
    )

    print(
        "df/dy = x^2 + 2y"
    )

    print()
    print(
        "Стационарная система "
        "даёт (-1,-0.5), "
        "точка лежит на границе."
    )

    print()
    print(
        "Границы:"
    )

    print(
        "x=-1: "
        "f=y^2+y+1 -> [0.75,7]"
    )

    print(
        "x=2: "
        "f=y^2+4y-2 -> [-5,10]"
    )

    print(
        "y=-1: "
        "f=-x^2-x+1 -> [-5,1.25]"
    )

    print(
        "y=2: "
        "f=2x^2-x+4 -> [3.875,10]"
    )

    print()
    print(
        "ran(f3,X)=",
        exact
    )

    print(
        f"mid={exact.mid:.6f}, "
        f"rad={exact.rad:.6f}, "
        f"wid={exact.wid:.6f}"
    )


    f_eir = (
        (X ** 2) * Y
        + Y ** 2
        - X
    )

    f_equiv = (
        X * (X * Y - 1)
        + Y ** 2
    )

    print()
    print(
        "Естественные интервальные "
        "расширения:"
    )

    print_result(
        "Исходная форма",
        f_eir,
        exact
    )

    print_result(
        "X(XY-1)+Y^2",
        f_equiv,
        exact
    )


    dfdx_X = (
        2 * X * Y
        - 1
    )

    # df/dy =
    #
    # X^2 + 2Y

    dfdy_X = (
        X ** 2
        + 2 * Y
    )

    print()
    print(
        "Интервальные частные производные:"
    )

    print(
        "df/dx(X) =",
        dfdx_X
    )

    print(
        "df/dy(X) =",
        dfdy_X
    )

    def Fmv(cx, cy):

        return (
            f(cx, cy)

            + dfdx_X
            * (X - cx)

            + dfdy_X
            * (Y - cy)
        )

    centers = [

        # середина бруса
        (
            X.mid,
            Y.mid
        ),

        (-1, -1),

        (2, 2)

    ]

    mv_results = {}

    print()
    print("F_mv:")

    for cx, cy in centers:

        result = Fmv(
            cx,
            cy
        )

        mv_results[
            (cx, cy)
        ] = result

        print_result(
            f"c=({cx:.2f},{cy:.2f})",
            result,
            exact
        )


    px = baumann_p(
        dfdx_X
    )

    py = baumann_p(
        dfdy_X
    )

    c_star = (

        X.mid
        - px * X.rad,

        Y.mid
        - py * Y.rad

    )

    c_star2 = (

        X.mid
        + px * X.rad,

        Y.mid
        + py * Y.rad

    )

    F_star = Fmv(
        *c_star
    )

    F_star2 = Fmv(
        *c_star2
    )

    F_bic = (
        F_star
        .intersect(F_star2)
    )

    print()
    print(
        "Форма Бауманна:"
    )

    print(
        "px =",
        px
    )

    print(
        "py =",
        py
    )

    print(
        "c_* =",
        c_star
    )

    print(
        "c^* =",
        c_star2
    )

    print(
        "F_mv(c_*) =",
        F_star
    )

    print(
        "F_mv(c^*) =",
        F_star2
    )

    print_result(
        "F_bic",
        F_bic,
        exact
    )



    def f_eir_box(
            X_box,
            Y_box
    ):

        return (
            (X_box ** 2)
            * Y_box
            + Y_box ** 2
            - X_box
        )

    x_mid = X.mid

    X_left = Interval(
        X.low,
        x_mid
    )

    X_right = Interval(
        x_mid,
        X.high
    )

    Fx1 = f_eir_box(
        X_left,
        Y
    )

    Fx2 = f_eir_box(
        X_right,
        Y
    )

    hull_x = Interval.hull(
        Fx1,
        Fx2
    )

    print()
    print(
        "Бисекция по x:"
    )

    print(
        "Подбрус 1:",
        X_left,
        "x",
        Y,
        "->",
        Fx1
    )

    print(
        "Подбрус 2:",
        X_right,
        "x",
        Y,
        "->",
        Fx2
    )

    print_result(
        "Оболочка по x",
        hull_x,
        exact
    )


    y_mid = Y.mid

    Y_low = Interval(
        Y.low,
        y_mid
    )

    Y_high = Interval(
        y_mid,
        Y.high
    )

    Fy1 = f_eir_box(
        X,
        Y_low
    )

    Fy2 = f_eir_box(
        X,
        Y_high
    )

    hull_y = Interval.hull(
        Fy1,
        Fy2
    )

    print()
    print(
        "Бисекция по y:"
    )

    print(
        "Подбрус 1:",
        X,
        "x",
        Y_low,
        "->",
        Fy1
    )

    print(
        "Подбрус 2:",
        X,
        "x",
        Y_high,
        "->",
        Fy2
    )

    print_result(
        "Оболочка по y",
        hull_y,
        exact
    )

    # df/dx = 2xy - 1
    #
    # на брусе:
    #
    # [-5,7]
    #
    # поэтому:
    #
    # L1 = 7

    L1 = 7

    # df/dy = x^2 + 2y
    #
    # на брусе:
    #
    # [-2,8]
    #
    # поэтому:
    #
    # L2 = 8

    L2 = 8

    lipschitz_limit = (
        L1 * X.rad
        + L2 * Y.rad
    )

    print()
    print(
        "Липшиц:"
    )

    print(
        "L1 =",
        L1
    )

    print(
        "L2 =",
        L2
    )

    print(
        "L1*radX1 + "
        "L2*radX2 =",
        lipschitz_limit
    )

    print(
        "rad исходного ЕИР =",
        f_eir.rad
    )


    xx = np.linspace(
        X.low,
        X.high,
        220
    )

    yy = np.linspace(
        Y.low,
        Y.high,
        220
    )

    XX, YY = np.meshgrid(
        xx,
        yy
    )

    ZZ = f(
        XX,
        YY
    )

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_surface(
        XX,
        YY,
        ZZ,
        alpha=0.85
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f3(x,y)")

    ax.set_title(
        "f3(x,y)=x²y+y²-x"
    )

    plt.tight_layout()

    plt.savefig(
        "fig_f3_surface_variant12.png",
        dpi=180
    )

    plt.close()


    plt.figure(
        figsize=(8, 6)
    )

    contours = plt.contour(
        XX,
        YY,
        ZZ,
        levels=18
    )

    plt.clabel(
        contours,
        inline=True,
        fontsize=8
    )

    rectangle = (
        plt.Rectangle(
            (
                X.low,
                Y.low
            ),

            X.wid,
            Y.wid,

            fill=False,
            linewidth=2,
            label="Брус X"
        )
    )

    plt.gca().add_patch(
        rectangle
    )


    plt.scatter(
        [2],
        [-1],
        s=60,
        label=
        "min: (2,-1), f=-5"
    )


    plt.scatter(
        [2],
        [2],
        s=60,
        label=
        "max: (2,2), f=10"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        "Линии уровня f3(x,y), "
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "fig_f3_contours_variant12.png",
        dpi=180
    )

    plt.close()


    print()
    print(
        "Сводные результаты f3:"
    )

    print_result(
        "Точная область",
        exact,
        exact
    )

    print_result(
        "ЕИР",
        f_eir,
        exact
    )

    print_result(
        "Эквивалентная форма",
        f_equiv,
        exact
    )

    print_result(
        "F_mv(mid X)",
        mv_results[
            (
                X.mid,
                Y.mid
            )
        ],
        exact
    )

    print_result(
        "Бауманн",
        F_bic,
        exact
    )

    print_result(
        "Бисекция по x",
        hull_x,
        exact
    )

    print_result(
        "Бисекция по y",
        hull_y,
        exact
    )


if __name__ == "__main__":

    solve_f1()

    solve_f2()

    solve_f3()


    print(
        "Созданы графики:"
    )

    print(
        "fig_f1.png"
    )

    print(
        "fig_f2_variant12.png"
    )

    print(
        "fig_f3_surface_variant12.png"
    )

    print(
        "fig_f3_contours_variant12.png"
    )