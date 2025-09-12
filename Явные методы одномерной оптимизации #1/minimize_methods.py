#!/usr/bin/env python3

import math
import argparse
import pandas as pd

def dichotomy(f, a, b, eps=1e-5, delta=None, max_iter=10000):
    if delta is None:
        delta = eps / 3
    iters = 0
    while (b - a) > 2 * eps and iters < max_iter:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        if f(x1) <= f(x2):
            b = x2
        else:
            a = x1
        iters += 1
    x_star = (a + b) / 2
    return x_star, f(x_star), iters

def golden_section(f, a, b, eps=1e-5, max_iter=10000):
    phi = (1 + 5 ** 0.5) / 2
    resphi = 2 - phi
    c = b - resphi * (b - a)
    d = a + resphi * (b - a)
    fc = f(c)
    fd = f(d)
    iters = 0
    while (b - a) > 2 * eps and iters < max_iter:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - resphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + resphi * (b - a)
            fd = f(d)
        iters += 1
    x_star = (a + b) / 2
    return x_star, f(x_star), iters

def f1(x):
    return float('inf') if x <= 0 else x*x - 3*x + x*math.log(x)
def f2(x):
    return math.log(1 + x*x) - math.sin(x)
def f3(x):
    return x**4 + math.exp(-x)
def f4(x):
    return (100 - x)**4
def f5(x):
    return float('inf') if x <= 0 else 2*x*x + 16/x

FUNCTIONS = {
    'f1': {'name': 'x^2 - 3x + x ln x, x∈[a,5], a>0', 'f': f1, 'interval': (0.1, 5.0)},
    'f2': {'name': 'ln(1 + x^2) - sin x, x∈[0, π/4]', 'f': f2, 'interval': (0.0, math.pi/4)},
    'f3': {'name': 'x^4 + e^{-x}, x∈[0,1]', 'f': f3, 'interval': (0.0, 1.0)},
    'f4': {'name': '(100 - x)^4, x∈[60,150]', 'f': f4, 'interval': (60.0, 150.0)},
    'f5': {'name': '2x^2 + 16/x, x∈[0.5,5]', 'f': f5, 'interval': (0.5, 5.0)},
}

def format_cell(x, y, eps, iters):
    return f"x*={x:.8f}; f(x*)={y:.8f}; eps={eps:.1e}; iters={iters}"

def run_one(func_key, method, eps=1e-5, a=None, b=None):
    meta = FUNCTIONS[func_key]
    f = meta['f']
    ia, ib = meta['interval']
    if a is None: a = ia
    if b is None: b = ib
    if method == 'dichotomy':
        return dichotomy(f, a, b, eps)
    elif method == 'golden':
        return golden_section(f, a, b, eps)
    else:
        raise ValueError("method must be 'dichotomy' or 'golden'")

def write_outputs(rows, eps, csv_path='results_minimization.csv', md_path='README.md'):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    readme = (
        '# Минимизация одномерных функций: дихотомия и золотое сечение\n\n'
        'Этот проект сравнивает методы **дихотомии** и **золотого сечения** на пяти функциях и сохраняет таблицу результатов в CSV и Markdown.\n\n'
        '## Методы\n\n'
        '- **Дихотомия.** На шаге выбираем две точки \n'
        '  $ x_1 = \\frac{a+b-\\delta}{2},\\quad x_2 = \\frac{a+b+\\delta}{2} $'
        '  сравниваем $f(x_1)$ и $f(x_2)$, сужаем $[a,b]$. Остановка при $b-a \\le 2\\varepsilon$.\n\n'
        '- **Золотое сечение.** Держим точки $d$ и $d$ в пропорции золотого сечения и сдвигаем интервал, пока\\n'
        '  $ b-a \\le 2\\varepsilon. $\n\n'
        '## Функции и интервалы\n\n'
        '1. $( f_1(x)=x^2-3x+x\\ln x,\\ x\\in[a,5],\\ a>0 ) $  \n'
        '2. $( f_2(x)=\\ln(1+x^2)-\\sin x,\\ x\\in[0,\\pi/4] ) $  \n'
        '3. $( f_3(x)=x^4+e^{-x},\\ x\\in[0,1] ) $  \n'
        '4. $( f_4(x)=(100-x)^4,\\ x\\in[60,150] ) $  \n'
        '5. $( f_5(x)=2x^2+\\tfrac{16}{x},\\ x\\in[0.5,5] ) $\n\n'
        f'## Результаты (ε = {eps:.1e})\n\n'
        'Формат ячейки: `x*; f(x*); eps; iters`\n\n'
        '| Функция | Дихотомия | Золотое сечение |\n|---|---|---|\n'
    )
    for r in rows:
        readme += f"| {r['Функция']} | {r['Дихотомия']} | {r['Золотое сечение']} |\n"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(readme)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', choices=list(FUNCTIONS.keys()))
    parser.add_argument('--method', choices=['dichotomy', 'golden'])
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--a', type=float)
    parser.add_argument('--b', type=float)
    args = parser.parse_args()

    if args.func and args.method:
        x, y, it = run_one(args.func, args.method, args.eps, args.a, args.b)
        print(format_cell(x, y, args.eps, it))
    else:
        print('| Функция | Дихотомия | Золотое сечение |')
        print('|---|---|---|')
        rows = []
        for key, meta in FUNCTIONS.items():
            a, b = meta['interval']
            xd, yd, itd = run_one(key, 'dichotomy', args.eps, a, b)
            xg, yg, itg = run_one(key, 'golden', args.eps, a, b)
            print('| ' + ' | '.join([
                f"{key}: {meta['name']}",
                format_cell(xd, yd, args.eps, itd),
                format_cell(xg, yg, args.eps, itg),
            ]) + ' |')
            rows.append({
                'Функция': f"{key}: {meta['name']}",
                'Дихотомия': format_cell(xd, yd, args.eps, itd),
                'Золотое сечение': format_cell(xg, yg, args.eps, itg),
            })
        write_outputs(rows, args.eps, csv_path='results_minimization.csv', md_path='results_minimization.md')
