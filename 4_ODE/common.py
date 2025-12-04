from copy import deepcopy

# --- Butcher-таблица метода 3/8 (RK 4(3/8))
rk4_38 = {
    "c": [0.0, 1/3, 2/3, 1.0],
    "A": [
        [0.0, 0.0, 0.0, 0.0],
        [1/3, 0.0, 0.0, 0.0],
        [-1/3, 1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0, 0.0]
    ],
    "b": [1/8, 3/8, 3/8, 1/8]
}


def rk_explicit_step(f, x, y, h, scheme):
    """
    Универсальный явный RK для векторных систем.
    f(x, y) -> list (вектор)
    y: list (вектор) — текущее значение
    Возвращает y_next: list
    """
    c = scheme["c"]
    A = scheme["A"]
    b = scheme["b"]
    s = len(c)

    # подготовить k: список векторов длины s, каждый длины len(y)
    n = len(y) if isinstance(y, list) else 1
    k = [[0.0]*n for _ in range(s)]

    for i_stage in range(s):
        # вычисляем y_i = y + h * sum_j A[i][j] * k[j]
        y_i = [val for val in y]  # копия
        for j in range(i_stage):
            coeff = A[i_stage][j]
            if coeff != 0.0:
                for idx in range(n):
                    y_i[idx] += h * coeff * k[j][idx]

        x_i = x + c[i_stage] * h
        f_val = f(x_i, y_i)
        # привести f_val к вектору
        if not isinstance(f_val, list) and not isinstance(f_val, tuple):
            f_val = [f_val]
        # записать k[i] = f_val
        for idx in range(n):
            k[i_stage][idx] = f_val[idx]

    # собрать y_next = y + h * sum_i b[i] * k[i]
    y_next = [val for val in y]
    for i_stage in range(s):
        bi = b[i_stage]
        if bi != 0.0:
            for idx in range(n):
                y_next[idx] += h * bi * k[i_stage][idx]

    return y_next


def solve_ode(f, x0, x1, y0, h, step_func, scheme=None):
    """
    Простая интеграция от x0 до x1 с шагом h, возвращает xs, ys, iter_count.
    ys — список векторов (list of lists)
    """
    x = x0
    y = deepcopy(y0) if isinstance(y0, list) else y0
    xs = [x]
    ys = [deepcopy(y)]
    iter_count = 0

    # безопасность: если h не делит сегмент, последний шаг будет меньше h
    while x < x1 - 1e-14:
        h_step = h
        if x + h_step > x1:
            h_step = x1 - x
        y_next = step_func(f, x, deepcopy(y), h_step, scheme)
        x += h_step
        y = deepcopy(y_next)
        xs.append(x)
        ys.append(deepcopy(y))
        iter_count += 1

    return xs, ys, iter_count
