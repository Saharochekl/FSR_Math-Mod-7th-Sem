"""
Метод ломаных (Пиявского–Шуберта) + классические одномерные методы (дихотомия, золотое сечение, Фибоначчи) для таблично заданных функций.
Читает таблицы из текущей директории, строит линейную интерполяцию, выполняет поиск минимума и сохраняет результаты в `out/` (графики в `out/plots/`).
Соответствует базовым правилам PEP 257 для docstring.
"""
import re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Атоматизация директорий
BASE = Path(__file__).resolve().parent
IN_DIR = BASE                  # Предполагается, что скрипт и инпуты в одной директории
OUT = BASE / "out"            # все результаты сюда
PLOTS = OUT / "plots"         # графики сюда
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


_num_re = re.compile(r'^[\s+-]?(\d+([.,]\d+)?)([eE][+-]?\d+)?\s*$')

def _read_table(path):
    """
    Читает табличную функцию из текстового/CSV файла и возвращает очищенные пары (x, y).

    Args:
        path (Path | str): путь к файлу с таблицей (разделитель `;` или `,`).

    Returns:
        pandas.DataFrame: столбцы `x`, `y` (float), отсортированные по `x` без дубликатов.

    Notes:
        * Если первый столбец распознаётся как дата/время, `x` переводится в секунды от первого значения.
        * Если даты нет, `x` читается как число; при невозможности — используется индексация.
        * В качестве `y` используется второй числовой столбец строки.
    """
    with open(path, 'r', errors='ignore') as f:
        lines = f.read().splitlines()
    # выбор разделителя по первой подходящей строке
    delim = None
    for ln in lines:
        if ';' in ln and ln.count(';') >= 1: delim = ';'; break
        if ',' in ln and ln.count(',') >= 1: delim = ','; break
    if delim is None: delim = ';'
    x_raw, y_raw = [], []
    for ln in lines:
        if delim not in ln: continue
        parts = [p.strip() for p in ln.split(delim)]
        if len(parts) < 2: continue
        ytok = parts[1].replace(' ', '').replace('\u00A0','').replace(',', '.')
        if not _num_re.match(ytok):  # не число во 2-м столбце → пропустить (заголовки и т.п.)
            continue
        x_raw.append(parts[0])
        y_raw.append(float(ytok))
    if not y_raw:
        raise ValueError("Нет числовых данных во 2-м столбце")
    # x: сначала попытка datetime
    xdt = pd.to_datetime(pd.Series(x_raw), errors='coerce', dayfirst=True)
    if xdt.notna().sum() > 0:
        base = xdt.dropna().iloc[0]
        x = (xdt - base).dt.total_seconds().ffill().bfill().astype(float).values
    else:
        # попытка как число
        xnum = []
        for s in x_raw:
            st = s.replace(' ', '').replace('\u00A0','').replace(',', '.')
            try: xnum.append(float(st))
            except: xnum.append(np.nan)
        if not np.isfinite(xnum).any():
            x = np.arange(len(y_raw), dtype=float)
        else:
            # где NaN — заменим равномерной сеткой
            x = np.array(xnum, dtype=float)
            mask = ~np.isfinite(x)
            if mask.any():
                x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    df = pd.DataFrame({"x": x, "y": y_raw}).dropna().astype(float)
    df = df.drop_duplicates(subset=["x"]).sort_values("x").reset_index(drop=True)
    return df

def make_interp_func(x, y):
    """
    Создаёт линейный интерполятор по табличным точкам и возвращает удобные границы.

    Args:
        x (array-like): узлы по оси X.
        y (array-like): значения функции в узлах.

    Returns:
        tuple: `(f, a, b, x_sorted, y_sorted)`, где `f(t)` — интерполирующая функция,
        `a = min(x)`, `b = max(x)`, а также отсортированные массивы `x_sorted`, `y_sorted` без дубликатов по X.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    idx = np.argsort(x); x = x[idx]; y = y[idx]
    x, uniq_idx = np.unique(x, return_index=True); y = y[uniq_idx]
    a, b = float(x[0]), float(x[-1])
    def f(t): return np.interp(np.asarray(t, float), x, y)
    return f, a, b, x, y

def estimate_L(x, y, r=1.3):
    """
    Оценка константы Липшица L по табличным данным.

    Args:
        x (array-like): узлы X.
        y (array-like): значения Y.
        r (float): коэффициент запаса к оценке (по умолчанию 1.3).

    Returns:
        float: `L = r * max |Δy/Δx|`.
    """
    dx = np.diff(x); dy = np.abs(np.diff(y))
    m = float(np.max(dy / np.maximum(dx, 1e-12))) if len(dx) else 1.0
    return r*m if m>0 else 1.0

def plot_broken_lines(f, a, b, L, xs, fs, tag="plot"):
    """
    Строит интерполированную `f(x)` и нижнюю оценку `g(x; x_i)` для узлов метода ломаных и сохраняет PNG.

    Args:
        f (callable): интерполирующая функция.
        a (float): левая граница интервала.
        b (float): правая граница интервала.
        L (float): оценка Липшица.
        xs (array-like): текущие узлы метода ломаных.
        fs (array-like): значения `f` в узлах `xs`.
        tag (str): суффикс имени файла графика.
    """
    grid = np.linspace(a, b, 800)
    fg = f(grid)
    G = np.max(fs.reshape(-1,1) - L*np.abs(grid.reshape(1,-1) - xs.reshape(-1,1)), axis=0)
    plt.figure(figsize=(6,4))
    plt.plot(grid, fg, label="f(x) интерп.")
    plt.plot(grid, G, label="g(x; x_i) ломаная")
    for xj in xs: plt.axvline(xj, alpha=0.25)
    plt.title(f"Метод ломаных: {tag}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / f"{tag}.png", dpi=150); plt.close()

def broken_lines_minimize(f, a, b, L, iters=25, plot_every=(1,5,10,20,25), tag="f"):
    """
    Метод ломаных (Пиявского–Шуберта) для глобального поиска минимума на отрезке при известной оценке Липшица.

    Args:
        f (callable): целевая функция (интерполяция по таблице).
        a (float): левая граница.
        b (float): правая граница.
        L (float): оценка константы Липшица.
        iters (int): число итераций.
        plot_every (tuple[int]): номера итераций, на которых рисовать графики.
        tag (str): базовый тег для имён файлов.

    Returns:
        dict: `{"x", "f", "evals", "xs", "fs", "hist"}` — точка минимума, значение,
        число вызовов `f`, финальные узлы и краткая история итераций.
    """
    xs = [a, b]; fs = [f(a), f(b)]; evals = 2
    order = np.argsort(xs); xs = [xs[i] for i in order]; fs = [fs[i] for i in order]
    hist = []
    for k in range(1, iters+1):
        xs_arr = np.array(xs); fs_arr = np.array(fs)
        dx = xs_arr[1:] - xs_arr[:-1]
        x_new = 0.5*(xs_arr[1:] + xs_arr[:-1]) - (fs_arr[1:] - fs_arr[:-1])/(2*L)
        R = 0.5*(fs_arr[1:] + fs_arr[:-1]) - L*dx/2.0
        i = int(np.argmax(R))
        xi, xip1 = xs[i], xs[i+1]
        xn = float(np.clip(x_new[i], xi, xip1))
        fn = float(f(xn)); evals += 1
        xs.insert(i+1, xn); fs.insert(i+1, fn)
        hist.append((k, xn, fn, float(np.max(R))))
        if k in plot_every:
            plot_broken_lines(f, a, b, L, np.array(xs), np.array(fs), tag=f"{tag}_iter{k}")
    best_idx = int(np.argmin(fs))
    return {"x": float(xs[best_idx]), "f": float(fs[best_idx]), "evals": evals, "xs": np.array(xs), "fs": np.array(fs), "hist": hist}

def dichotomy(f, a, b, eps=1e-3, delta=1e-5):
    """
    Метод дихотомии для унимодальной минимизации на [a, b].

    Args:
        f (callable): целевая функция.
        a (float): левая граница.
        b (float): правая граница.
        eps (float): требуемая точность по длине интервала.
        delta (float): половина смещения от середины при сравнении.

    Returns:
        tuple: `(xm, f(xm), evals)` — приблизительный минимум, значение и число вызовов `f`.
    """
    evals = 0
    while b - a > eps:
        m = 0.5*(a+b); x1, x2 = m - delta, m + delta
        f1, f2 = f(x1), f(x2); evals += 2
        if f1 < f2: b = x2
        else: a = x1
    xm = 0.5*(a+b)
    return xm, float(f(xm)), evals

def golden(f, a, b, eps=1e-3):
    """
    Метод золотого сечения для унимодальной минимизации на [a, b].

    Args:
        f (callable): целевая функция.
        a (float): левая граница.
        b (float): правая граница.
        eps (float): требуемая точность.

    Returns:
        tuple: `(xm, f(xm), evals)`.
    """
    gr = (math.sqrt(5) - 1)/2
    x1 = b - gr*(b-a); x2 = a + gr*(b-a)
    f1, f2 = f(x1), f(x2); evals = 2
    while b - a > eps:
        if f1 > f2:
            a = x1; x1 = x2; f1 = f2
            x2 = a + gr*(b-a); f2 = f(x2); evals += 1
        else:
            b = x2; x2 = x1; f2 = f1
            x1 = b - gr*(b-a); f1 = f(x1); evals += 1
    xm = 0.5*(a+b)
    return xm, float(f(xm)), evals

def fibonacci_search(f, a, b, eps=1e-3):
    """
    Метод поиска по Фибоначчи для унимодальной минимизации на [a, b].

    Args:
        f (callable): целевая функция.
        a (float): левая граница.
        b (float): правая граница.
        eps (float): требуемая точность.

    Returns:
        tuple: `(xm, f(xm), evals)`.
    """
    F = [1, 1]
    while F[-1] < (b - a)/eps:
        F.append(F[-1] + F[-2])
    n = len(F) - 1
    x1 = a + (F[n-2]/F[n])*(b - a)
    x2 = a + (F[n-1]/F[n])*(b - a)
    f1, f2, evals = f(x1), f(x2), 2
    for k in range(1, n-1):
        if f1 > f2:
            a = x1; x1 = x2; f1 = f2
            x2 = a + (F[n-k-1]/F[n-k])*(b - a); f2 = f(x2); evals += 1
        else:
            b = x2; x2 = x1; f2 = f1
            x1 = a + (F[n-k-2]/F[n-k])*(b - a); f1 = f(x1); evals += 1
    xm = 0.5*(a+b)
    return xm, float(f(xm)), evals

names = {
    "angle-func.txt","angular-velocity-function.txt","distance_txt.txt","satellite-facility-dist.txt",
    "angle-func.csv","angular-velocity-function.csv","distatnce_csv.csv","satellite-facility-dist.csv",
}
files = list(IN_DIR.glob("*.txt")) + list(IN_DIR.glob("*.csv"))
found = [p for p in files if p.name in names and p.is_file()]

rows = []
for path in found:
    base = path.stem
    data = _read_table(path)
    f, a, b, x_tab, y_tab = make_interp_func(data["x"], data["y"])
    L = estimate_L(x_tab, y_tab, r=1.3)
    bl = broken_lines_minimize(f, a, b, L, iters=25, plot_every=(1,5,10,20,25), tag=f"{base}")
    xd, fd, ed = dichotomy(f, a, b, eps=(b-a)*1e-4)
    xg, fg, eg = golden(f, a, b, eps=(b-a)*1e-4)
    xf, ff, ef = fibonacci_search(f, a, b, eps=(b-a)*1e-4)
    pd.DataFrame({"x": bl["xs"], "f": bl["fs"]}).to_csv(OUT / f"{base}__broken_points.csv", index=False)
    with open(OUT / f"{base}__methods.txt", "w", encoding="utf-8") as w:
        w.write(f"Файл: {path.name}\n")
        w.write(f"[a,b]=[{a:.6f},{b:.6f}], L≈{L:.6f}\n")
        w.write(f"Метод ломаных: x*={bl['x']:.6f}, f*={bl['f']:.9f}, evals={bl['evals']}\n")
        w.write(f"Дихотомия:     x*={xd:.6f}, f*={fd:.9f}, evals={ed}\n")
        w.write(f"Золотое сеч.:  x*={xg:.6f}, f*={fg:.9f}, evals={eg}\n")
        w.write(f"Фибоначчи:     x*={xf:.6f}, f*={ff:.9f}, evals={ef}\n")
    rows.append({
        "file": path.name, "a": a, "b": b, "L": L,
        "x_broken": bl["x"], "f_broken": bl["f"], "evals_broken": bl["evals"],
        "x_dicho": xd, "f_dicho": fd, "evals_dicho": ed,
        "x_golden": xg, "f_golden": fg, "evals_golden": eg,
        "x_fib": xf, "f_fib": ff, "evals_fib": ef,
        "points_saved": f"{base}__broken_points.csv",
        "methods_saved": f"{base}__methods.txt",
        "plots_prefix": f"plots/{base}_iter*.png"
    })

summary = pd.DataFrame(rows)
summary_path = OUT / "results_summary.csv"
summary.to_csv(summary_path, index=False)


print("OK:", summary_path)
print("Папка с графиками:", PLOTS)