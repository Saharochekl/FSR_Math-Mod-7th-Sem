import pandas as pd, numpy as np, io, os, heapq
from typing import Tuple, Dict, List, Any

DATA_FILES = [
    "angle-func.txt",
    "angular-velocity-function.txt",
    "distance_txt.txt",
    "satellite-facility-dist.txt",
]

def read_text_any_encoding(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "iso-8859-1", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        return f.read()

def find_header_row(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.strip().split(";")]
        if len(parts) >= 2 and ("Время" in parts[0] or "Time" in parts[0]):
            return i
    for i, line in enumerate(lines):
        if line.count(";") >= 1:
            return i
    return 0

def read_mide_table(path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    text = read_text_any_encoding(path)
    lines = text.splitlines()
    hdr = find_header_row(text)
    csv_text = "\n".join(lines[hdr:])
    df = pd.read_csv(io.StringIO(csv_text), sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    time_col = next((c for c in df.columns if "Время" in c or "Time" in c), df.columns[0])
    t = pd.to_datetime(df[time_col].astype(str), dayfirst=True, errors="coerce")
    if t.isna().all():
        t_sec = np.arange(len(df), dtype=float)
    else:
        t_sec = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    num = df.drop(columns=[time_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all")
    mask = ~num.isna().any(axis=1) & ~pd.isna(t_sec)
    return t_sec[mask.values], num[mask.values].reset_index(drop=True)

def estimate_L(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x); dy = np.abs(np.diff(y))
    mask = dx > 0
    if not np.any(mask): return 0.0
    slopes = dy[mask] / dx[mask]
    if len(slopes) == 0: return 0.0
    q95 = np.quantile(slopes, 0.95)
    smax = np.max(slopes)
    return float(max(q95, 0.9 * smax) + 1e-12)

def seq_enum(x: np.ndarray, y: np.ndarray) -> Tuple[int,float]:
    i = int(np.argmin(y)); return i, float(y[i])

def interval_lb(y_i, y_j, x_i, x_j, L) -> Tuple[float,float]:
    x_c = (x_i + x_j + (y_j - y_i)/L) / 2.0
    r = max(y_i - L*(x_c - x_i), y_j - L*(x_j - x_c))
    return r, x_c

def nearest_interior_index(x: np.ndarray, x_target: float, i: int, j: int) -> int:
    if j <= i + 1: return -1
    left, right = i + 1, j - 1
    return left + int(np.argmin(np.abs(x[left:right+1] - x_target)))

def bb_lipschitz_discrete(x: np.ndarray, y: np.ndarray, L: float, tol: float = 0.0) -> Tuple[int,float,int]:
    n = len(x)
    if n == 0: return -1, np.nan, 0
    evaluated = np.zeros(n, dtype=bool)
    evaluated[0] = evaluated[-1] = True
    f_best = min(y[0], y[-1]); i_best = int(0 if y[0] <= y[-1] else n-1)
    heap = []
    r, x_c = interval_lb(y[0], y[-1], x[0], x[-1], L)
    k = nearest_interior_index(x, x_c, 0, n-1)
    if k != -1:
        heapq.heappush(heap, (r, 0, n-1, k, x_c))
    evals = 2
    while heap:
        r_ij, i, j, k, x_c = heapq.heappop(heap)
        if r_ij >= f_best - tol: break
        if evaluated[k]: continue
        evaluated[k] = True; f_k = y[k]; evals += 1
        if f_k < f_best: f_best, i_best = f_k, k
        if k > i + 1:
            r_l, x_cl = interval_lb(y[i], y[k], x[i], x[k], L)
            k_l = nearest_interior_index(x, x_cl, i, k)
            if k_l != -1: heapq.heappush(heap, (r_l, i, k, k_l, x_cl))
        if j > k + 1:
            r_r, x_cr = interval_lb(y[k], y[j], x[k], x[j], L)
            k_r = nearest_interior_index(x, x_cr, k, j)
            if k_r != -1: heapq.heappush(heap, (r_r, k, j, k_r, x_cr))
    return i_best, float(f_best), evals

def process_file(path: str) -> pd.DataFrame:
    x, df = read_mide_table(path)
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        y = df[col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x2, y2 = x[mask], y[mask]
        if len(x2) < 3: continue
        order = np.argsort(x2); x2, y2 = x2[order], y2[order]
        i_seq, f_seq = seq_enum(x2, y2)
        L = estimate_L(x2, y2)
        i_bb, f_bb, used = bb_lipschitz_discrete(x2, y2, L)
        n = len(x2)
        rows.append(dict(
            file=os.path.basename(path), column=col, N_total=n, L_est=L,
            seq_argmin_index=i_seq, seq_argmin_time_s=x2[i_seq], seq_min_value=f_seq,
            bb_argmin_index=i_bb, bb_argmin_time_s=x2[i_bb], bb_min_value=f_bb,
            evals_used_bb=used, eval_savings_pct=100.0*(1.0 - used/n),
            exact_match=bool(i_seq==i_bb and abs(f_seq-f_bb)<=1e-9)
        ))
    return pd.DataFrame(rows)

def main():
    out = []
    for f in DATA_FILES:
        if os.path.exists(f):
            out.append(process_file(f))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    res.to_csv("enumeration_summary.csv", index=False)
    print(res)

if __name__ == "__main__":
    main()