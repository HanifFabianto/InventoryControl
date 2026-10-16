"""
Inventory Control Dashboard (Streamlit)

Jalankan:  streamlit run app.py

Isi file:
  1. Konstanta dan fungsi bantu
  2. Data: membaca CSV, validasi, statistik per item
  3. Perhitungan analitis: EOQ, safety stock, ROP, biaya
  4. Simulasi Monte Carlo: kebijakan (Q, R) dan (P, S)
  5. Komponen tampilan (satu fungsi per tab)
  6. main()

Semua parameter perhitungan memakai satuan PERIODE data (satu baris = satu periode,
sesuai pilihan frekuensi di sidebar). Biaya tahunan memakai periods_per_year.
"""

import hashlib
import io
import json
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 1. Konstanta dan fungsi bantu
# ---------------------------------------------------------------------------
ACCENT = "#1F5FA8"
INK = "#2B2F36"
MUTED = "#8A8F98"
RULE = "#D9DCE1"
NONE = "(tidak ada)"
NORM = NormalDist()

FREQUENCIES = {"Harian": 365.0, "Mingguan": 52.0, "Bulanan": 12.0, "Kuartalan": 4.0, "Kustom": None}
SRC_NORMAL = "Normal (dipotong di 0)"
SRC_BOOT = "Bootstrap dari data historis"

ITEM_HINTS = {"store id", "product id", "sku", "item", "product", "store"}
HINTS = {
    "date": ["date", "tanggal", "period", "periode", "week", "month"],
    "demand": ["demand", "permintaan"],
    "inv": ["inventory level", "inventory", "stock level", "on hand", "onhand"],
    "sold": ["units sold", "sold", "sales"],
    "ordered": ["units ordered", "ordered"],
    "unit_cost": ["unit cost", "unit_cost", "cogs", "purchase cost", "biaya satuan"],
    "lead_time": ["lead time", "leadtime", "lead_time"],
    "order_cost": ["ordering cost", "order cost", "ordering_cost"],
    "hold_cost": ["holding cost", "holding_cost"],
    "penalty": ["stockout cost", "stockout penalty", "shortage cost", "penalty"],
}


def z_of(service_level):
    return NORM.inv_cdf(service_level)


def unit_loss(z):
    """Unit normal loss function G(z) = phi(z) - z * (1 - Phi(z))."""
    return NORM.pdf(z) - z * (1.0 - NORM.cdf(z))


def finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def fmt(x, nd=2, pct=False):
    if not finite(x):
        return "—"
    v = float(x)
    return f"{v * 100:,.{nd}f}%" if pct else f"{v:,.{nd}f}"


def guess_index(cols, hints):
    """Indeks (dalam [NONE] + cols) dari kolom yang namanya cocok dengan petunjuk; 0 jika tidak ada."""
    low = [c.lower().strip() for c in cols]
    for h in hints:
        for i, c in enumerate(low):
            if c == h:
                return i + 1
    for h in hints:
        for i, c in enumerate(low):
            if h in c:
                return i + 1
    return 0


def signature(*parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def style(fig, title=None, height=340):
    fig.update_layout(
        template="simple_white",
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(color=INK),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Membaca CSV...")
def load_csv(content, sep):
    return pd.read_csv(io.BytesIO(content), sep=sep)


@st.cache_data(show_spinner="Memvalidasi data...")
def prepare_data(raw, m, item_cols, missing_rule):
    """Bentuk tabel kerja dengan nama kolom baku. Mengembalikan (tabel, catatan).

    Kolom hasil: demand, item, [date, inv, sold, ordered, unit_cost, order_cost,
    hold_cost, penalty, lead_time] sesuai pemetaan. Melempar ValueError jika data tidak layak.
    """
    notes = []
    d = pd.DataFrame(index=raw.index)

    original = raw[m["demand"]]
    dem = pd.to_numeric(original, errors="coerce")
    bad = int((dem.isna() & original.notna()).sum())
    if bad:
        notes.append(f"{bad:,} nilai pada kolom demand bukan angka dan diperlakukan sebagai kosong.")
    d["demand"] = dem

    if item_cols:
        key = raw[item_cols[0]].astype(str)
        for c in item_cols[1:]:
            key = key + " | " + raw[c].astype(str)
        d["item"] = key
    else:
        d["item"] = "Seluruh data"

    if m.get("date"):
        dates = pd.to_datetime(raw[m["date"]], errors="coerce")
        if dates.isna().all():
            raise ValueError(f"Kolom tanggal '{m['date']}' tidak dapat dibaca sebagai tanggal.")
        d["date"] = dates

    for role in ("inv", "sold", "ordered", "unit_cost", "order_cost", "hold_cost", "penalty", "lead_time"):
        if m.get(role):
            col = pd.to_numeric(raw[m[role]], errors="coerce")
            bad = int((col.isna() & raw[m[role]].notna()).sum())
            if bad:
                notes.append(f"{bad:,} nilai pada kolom '{m[role]}' bukan angka dan diperlakukan sebagai kosong.")
            d[role] = col

    if "date" in d:
        n_bad_date = int(d["date"].isna().sum())
        if n_bad_date:
            notes.append(f"{n_bad_date:,} baris dengan tanggal tidak valid dibuang.")
            d = d[d["date"].notna()]

    n_missing = int(d["demand"].isna().sum())
    if n_missing:
        if missing_rule == "Buang baris":
            notes.append(f"{n_missing:,} baris dengan demand kosong dibuang.")
            d = d[d["demand"].notna()]
        else:
            notes.append(f"{n_missing:,} demand kosong diisi 0 (asumsi: tidak ada permintaan).")
            d = d.assign(demand=d["demand"].fillna(0.0))

    n_neg = int((d["demand"] < 0).sum())
    if n_neg:
        notes.append(f"{n_neg:,} baris dengan demand negatif dibuang (demand harus >= 0).")
        d = d[d["demand"] >= 0]

    if d.empty:
        raise ValueError("Tidak ada baris valid yang tersisa setelah validasi. Periksa kolom demand dan tanggal.")

    if "date" in d:
        d = d.sort_values(["item", "date"], kind="stable")
        dup = int(d.duplicated(["item", "date"]).sum())
        if dup:
            notes.append(
                f"{dup:,} baris memiliki kombinasi (item, tanggal) yang sama. Time series per item menjadi tidak "
                "tunggal; tambahkan kolom identitas item (mis. Store ID) atau agregasi data lebih dulu."
            )
    return d.reset_index(drop=True), notes


def infer_gap_days(dates):
    u = np.sort(pd.Series(dates).dropna().unique())
    if len(u) < 2:
        return None
    gaps = np.diff(u).astype("timedelta64[D]").astype(float)
    return float(np.median(gaps))


def freq_default(gap):
    if gap is None:
        return 0
    if gap <= 1.5:
        return 0
    if 5.5 <= gap <= 8.5:
        return 1
    if 27 <= gap <= 32:
        return 2
    if 85 <= gap <= 95:
        return 3
    return 4


def item_stats_table(d):
    """Statistik per item. Kolom: n, mu, sd, med, dmin, dmax, c, S, h_col, pen, lt_mean, lt_sd."""
    g = d.groupby("item", sort=True)
    t = g["demand"].agg(n="count", mu="mean", sd="std", med="median", dmin="min", dmax="max")
    for src, dst in (("unit_cost", "c"), ("order_cost", "S"), ("hold_cost", "h_col"),
                     ("penalty", "pen"), ("lead_time", "lt_mean")):
        t[dst] = g[src].mean() if src in d else np.nan
    t["lt_sd"] = g["lead_time"].std(ddof=1) if "lead_time" in d else np.nan
    t["cv"] = t["sd"] / t["mu"].where(t["mu"] > 0)
    return t


def params_from_stats(r, cfg):
    """Parameter satu item dalam satuan PERIODE. Nilai manual dipakai bila kolom data tidak tersedia."""
    ppy = cfg["ppy"]
    pdays = 365.0 / ppy
    src = {}

    mu = float(r["mu"])
    if cfg["sd_mode"] == "manual":
        sd, src["sd"] = float(cfg["sd_manual"]), "input manual"
    else:
        sd, src["sd"] = float(r["sd"]), "data historis"

    if finite(r["c"]):
        c, src["c"] = float(r["c"]), "kolom data"
    else:
        c, src["c"] = float(cfg["c_manual"]), "input manual"
    if finite(r["S"]):
        S, src["S"] = float(r["S"]), "kolom data"
    else:
        S, src["S"] = float(cfg["S_manual"]), "input manual"
    if finite(r["pen"]):
        pen, src["pen"] = float(r["pen"]), "kolom data"
    else:
        pen, src["pen"] = float(cfg["pen_manual"]), "input manual"
    if finite(r["h_col"]):
        h, src["h"] = float(r["h_col"]), "kolom data"
    elif cfg["h_mode"] == "rate":
        h, src["h"] = float(cfg["h_rate"]) * c, "persentase x unit cost"
    else:
        h, src["h"] = float(cfg["h_fixed"]), "input manual"

    if finite(r["lt_mean"]):
        L_raw, src["L"] = float(r["lt_mean"]), "kolom data"
        sL_raw = float(r["lt_sd"]) if finite(r["lt_sd"]) else 0.0
    else:
        L_raw, sL_raw, src["L"] = float(cfg["lt_mean"]), float(cfg["lt_sd"]), "input manual"
    if cfg["lt_unit"] == "Hari":
        L, sL = L_raw / pdays, sL_raw / pdays
    else:
        L, sL = L_raw, sL_raw

    return dict(n=int(r["n"]), mu=mu, sd=sd, L=L, sL=sL, c=c, S=S, h=h, pen=pen,
                ppy=ppy, pdays=pdays, src=src)


def observed_table(d, cfg, method):
    """Metrik OBSERVASI per item dari kolom persediaan. method: 'stock' atau 'sold'."""
    need = ["inv"] + (["sold"] if method == "sold" else [])
    o = d.dropna(subset=need).copy()
    mu = o.groupby("item")["demand"].transform("mean")
    if method == "sold":
        unmet = (o["demand"] - o["sold"]).clip(lower=0)
    else:
        unmet = (o["demand"] - o["inv"]).clip(lower=0)
    o["unmet"] = unmet
    o["served"] = o["demand"] - unmet
    o["short"] = (unmet > 0).astype(float)
    o["excess"] = (o["inv"] - cfg["k_cover"] * mu).clip(lower=0)
    o["over"] = (o["excess"] > 0).astype(float)
    if "ordered" in o:
        o["order_event"] = (o["ordered"] > 0).astype(float)
    g = o.groupby("item")
    t = g.agg(n=("demand", "count"), dem=("demand", "sum"), unmet=("unmet", "sum"), served=("served", "sum"),
              stockout_freq=("short", "mean"), avg_inv=("inv", "mean"), over_freq=("over", "mean"),
              avg_excess=("excess", "mean"))
    if "ordered" in o:
        t["n_orders"] = g["order_event"].sum()
        t["ordered_units"] = g["ordered"].sum()
    t["fill_rate"] = 1 - t["unmet"] / t["dem"].where(t["dem"] > 0)
    t["turnover"] = (t["served"] / t["n"] * cfg["ppy"]) / t["avg_inv"].where(t["avg_inv"] > 0)
    t["dio"] = 365.0 / t["turnover"].where(t["turnover"] > 0)
    return t


# ---------------------------------------------------------------------------
# 3. Perhitungan analitis
# ---------------------------------------------------------------------------
def eoq(D, S, h):
    if not (D > 0 and S > 0 and h > 0):
        return np.nan
    return float(np.sqrt(2.0 * D * S / h))


def lead_time_demand_sd(L, sd, mu, sL):
    """sqrt(L * sd^2 + mu^2 * sL^2); asumsi demand antar-periode dan lead time saling independen."""
    return float(np.sqrt(L * sd ** 2 + (mu * sL) ** 2))


def analytic_policy(p, z):
    D = p["mu"] * p["ppy"]
    Q = eoq(D, p["S"], p["h"])
    sig = lead_time_demand_sd(p["L"], p["sd"], p["mu"], p["sL"])
    ss = z * sig
    return dict(D=D, Q=Q, sigma_lt=sig, SS=ss, ROP=p["mu"] * p["L"] + ss)


def periodic_policy(p, z, P):
    """Order-up-to level S untuk review periodik tiap P periode."""
    sig = float(np.sqrt((P + p["L"]) * p["sd"] ** 2 + (p["mu"] * p["sL"]) ** 2))
    return dict(P=P, S_level=p["mu"] * (P + p["L"]) + z * sig, sigma=sig)


def analytic_costs(p, pol, z):
    """Biaya tahunan aproksimasi untuk kebijakan (Q, R) dengan SS di atas. Semua per tahun."""
    Q, D = pol["Q"], pol["D"]
    short_cycle = pol["sigma_lt"] * unit_loss(z)           # ekspektasi unit kurang per siklus
    avg_inv = Q / 2.0 + pol["SS"]
    out = dict(
        ordering=p["S"] * D / Q,
        holding=p["h"] * avg_inv,
        stockout=p["pen"] * D / Q * short_cycle,
        purchase=p["c"] * D,
        avg_inv=avg_inv,
        fill_rate=float(np.clip(1.0 - short_cycle / Q, 0.0, 1.0)),
        turnover=D / avg_inv if avg_inv > 0 else np.nan,
    )
    out["total"] = out["ordering"] + out["holding"] + out["stockout"]
    out["dio"] = 365.0 / out["turnover"] if finite(out["turnover"]) and out["turnover"] > 0 else np.nan
    return out


def apply_multipliers(p, m):
    """Skenario: m berisi kunci mean, std, lt, h, S, pen (default 1)."""
    q = dict(p)
    mm, ms, ml = m.get("mean", 1.0), m.get("std", 1.0), m.get("lt", 1.0)
    q["mu"] = p["mu"] * mm
    q["sd"] = p["sd"] * mm * ms
    q["L"] = p["L"] * ml
    q["sL"] = p["sL"] * ml
    q["h"] = p["h"] * m.get("h", 1.0)
    q["S"] = p["S"] * m.get("S", 1.0)
    q["pen"] = p["pen"] * m.get("pen", 1.0)
    return q


def policy_table(stats, cfg, z):
    rows = []
    for item, r in stats.iterrows():
        p = params_from_stats(r, cfg)
        pol = analytic_policy(p, z)
        rows.append({
            "Item": item, "n periode": p["n"], "Demand rata-rata/periode": p["mu"], "Std demand/periode": p["sd"],
            "D (unit/tahun)": pol["D"], "S": p["S"], "H": p["h"], "EOQ": pol["Q"],
            "Pesanan/tahun": pol["D"] / pol["Q"] if finite(pol["Q"]) else np.nan,
            "Safety stock": pol["SS"], "ROP": pol["ROP"],
        })
    return pd.DataFrame(rows).set_index("Item")


# ---------------------------------------------------------------------------
# 4. Simulasi Monte Carlo
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(policy, draw, mult, cost, ppy, hist, n_rep, horizon, warmup, seed,
                   source, backorder, k_cover, init_stock):
    """Simulasi n_rep replikasi paralel.

    policy : ("QR", Q, R) atau ("PS", P, S)
    draw   : (mu, sd, L, sL) parameter dasar untuk membangkitkan demand dan lead time (satuan periode)
    mult   : (m_mean, m_std, m_lt) pengali skenario, diterapkan pada angka acak yang SAMA (common random numbers)
    cost   : (S, h, pen, c) biaya per pesanan, per unit per tahun, per unit kekurangan, per unit pembelian

    Urutan kejadian tiap periode t:
      1) penerimaan barang yang jatuh tempo di awal periode t
      2) demand terjadi; dipenuhi dari stok tersedia
      3) kekurangan: lost sales (hilang) atau backorder (dicatat sebagai stok negatif)
      4) di akhir periode, kebijakan memutuskan pesanan berdasarkan inventory position
         (on-hand - backorder + on-order); pesanan tiba di awal periode t + L + 1,
         sehingga L periode demand terjadi selama lead time.
    """
    mu, sd, L0, sL0 = draw
    m_mean, m_std, m_lt = mult
    S_cost, h, pen, c = cost
    T = int(warmup + horizon)
    rng = np.random.default_rng(seed)

    if source == SRC_NORMAL:
        base = mu + sd * rng.standard_normal((n_rep, T))
    else:
        base = rng.choice(np.asarray(hist, float), size=(n_rep, T), replace=True)
    zL = rng.standard_normal((n_rep, T))

    demand = np.clip(m_mean * (mu + m_std * (base - mu)), 0.0, None)
    Lall = np.maximum(0, np.rint(m_lt * (L0 + sL0 * zL))).astype(int)   # lead time dibulatkan ke periode utuh
    thr = k_cover * mu * m_mean                                          # ambang overstock (unit)

    rows = np.arange(n_rep)
    arrivals = np.zeros((n_rep, T + int(Lall.max()) + 2))
    IL = np.full(n_rep, float(init_stock))
    on_order = np.zeros(n_rep)
    flag = np.zeros(n_rep, dtype=bool)
    keys = ("dem", "unmet", "so_events", "inv", "over_events", "excess", "orders", "received", "cyc_tot", "cyc_so")
    acc = {k: np.zeros(n_rep) for k in keys}
    path_inv, path_ord = np.zeros(T), np.zeros(T)
    kind, a, b = policy
    if kind == "PS":
        a = max(1, int(round(a)))

    for t in range(T):
        rec = arrivals[:, t].copy()
        got = rec > 0
        IL = IL + rec
        on_order = on_order - rec
        measure = t >= warmup
        if t == warmup:
            flag[:] = False
        if measure:
            acc["cyc_tot"] += got                       # penerimaan = akhir satu siklus pemesanan
            acc["cyc_so"] += got & flag
            flag &= ~got
            acc["received"] += rec

        start_on = np.maximum(IL, 0.0)
        d = demand[:, t]
        served = np.minimum(d, start_on)
        unmet = d - served
        IL = IL - (d if backorder else served)
        end_on = np.maximum(IL, 0.0)
        short = unmet > 1e-9
        if measure:
            acc["dem"] += d
            acc["unmet"] += unmet
            acc["so_events"] += short
            flag |= short
            acc["inv"] += 0.5 * (start_on + end_on)      # rata-rata awal dan akhir periode
            acc["over_events"] += end_on > thr
            acc["excess"] += np.maximum(end_on - thr, 0.0)

        ip = IL + on_order
        if kind == "QR":
            k = np.floor((b - ip) / a) + 1.0               # kelipatan Q agar IP kembali di atas R
            qty = np.where(ip <= b, k * a, 0.0)
        else:
            qty = np.maximum(b - ip, 0.0) if (t + 1) % a == 0 else np.zeros(n_rep)
        placed = qty > 1e-9
        path_inv[t], path_ord[t] = end_on[0], qty[0]
        if placed.any():
            idx = t + Lall[placed, t] + 1
            np.add.at(arrivals, (rows[placed], idx), qty[placed])
            on_order = on_order + qty
            if measure:
                acc["orders"] += placed

    H = float(horizon)
    years = H / ppy
    avg_inv = acc["inv"] / H
    fulfilled = acc["dem"] - acc["unmet"]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = dict(
            ordering=S_cost * acc["orders"] / years,
            holding=h * avg_inv,
            stockout=pen * acc["unmet"] / years,
            purchase=c * acc["received"] / years,
            avg_inv=avg_inv,
            fill_rate=1.0 - acc["unmet"] / np.maximum(acc["dem"], 1e-12),
            stockout_freq=acc["so_events"] / H,
            unmet_per_year=acc["unmet"] / years,
            over_freq=acc["over_events"] / H,
            avg_excess=acc["excess"] / H,
            orders_per_year=acc["orders"] / years,
            turnover=(fulfilled / years) / np.where(avg_inv > 0, avg_inv, np.nan),
            csl=np.where(acc["cyc_tot"] > 0, 1.0 - acc["cyc_so"] / np.maximum(acc["cyc_tot"], 1.0), np.nan),
            cyc_tot=acc["cyc_tot"], cyc_so=acc["cyc_so"],
        )
        res["dio"] = 365.0 / res["turnover"]
    res["total"] = res["ordering"] + res["holding"] + res["stockout"]
    res["path"] = dict(inv=path_inv, order=path_ord, warmup=int(warmup))
    return res


def sim_means(res):
    """Rata-rata antar replikasi. CSL dihitung pooled: 1 - total siklus stockout / total siklus."""
    keys = ["total", "ordering", "holding", "stockout", "purchase", "fill_rate", "stockout_freq", "unmet_per_year",
            "avg_inv", "over_freq", "avg_excess", "orders_per_year", "turnover", "dio"]
    out = {k: float(np.nanmean(res[k])) for k in keys}
    tot = res["cyc_tot"].sum()
    out["csl"] = float(1.0 - res["cyc_so"].sum() / tot) if tot > 0 else np.nan
    return out


SIM_ROWS = [
    ("total", "Total biaya relevan / tahun", 2, False),
    ("ordering", "Biaya pemesanan / tahun", 2, False),
    ("holding", "Biaya penyimpanan / tahun", 2, False),
    ("stockout", "Biaya stockout / tahun", 2, False),
    ("purchase", "Biaya pembelian / tahun (di luar total)", 2, False),
    ("fill_rate", "Fill rate", 2, True),
    ("csl", "Cycle service level", 2, True),
    ("stockout_freq", "Proporsi periode dengan stockout", 2, True),
    ("unmet_per_year", "Unit tidak terpenuhi / tahun", 2, False),
    ("avg_inv", "Rata-rata persediaan (unit)", 2, False),
    ("over_freq", "Proporsi periode overstock", 2, True),
    ("avg_excess", "Rata-rata kelebihan di atas ambang (unit)", 2, False),
    ("orders_per_year", "Jumlah pesanan / tahun", 2, False),
    ("turnover", "Inventory turnover / tahun", 2, False),
    ("dio", "DIO (hari)", 1, False),
]


def sim_summary_table(res):
    means = sim_means(res)
    rows = []
    for key, label, nd, pct in SIM_ROWS:
        v = res[key]
        lo, hi = (np.nanpercentile(v, 5), np.nanpercentile(v, 95)) if np.isfinite(v).any() else (np.nan, np.nan)
        rows.append({"Metrik": label, "Rata-rata": fmt(means[key], nd, pct),
                     "P5": fmt(lo, nd, pct), "P95": fmt(hi, nd, pct)})
    return pd.DataFrame(rows).set_index("Metrik")


def simulate_policy(ctx, policy, mult=(1.0, 1.0, 1.0), cost=None):
    """Panggil simulasi dengan pengaturan dari sidebar. Seed sama => angka acak sama antar skenario."""
    p, s = ctx["p"], ctx["sim"]
    cost = cost or (p["S"], p["h"], p["pen"], p["c"])
    return run_simulation(
        policy, (p["mu"], p["sd"], p["L"], p["sL"]), tuple(mult), tuple(cost), p["ppy"],
        ctx["hist"], s["n_rep"], s["horizon"], s["warmup"], s["seed"], s["source"], s["backorder"],
        ctx["cfg"]["k_cover"], ctx["init_stock"],
    )


def sim_blockers(p, pol, sim):
    msgs = []
    if not finite(p["mu"]) or p["mu"] <= 0:
        msgs.append("Rata-rata demand item ini <= 0; simulasi tidak bermakna.")
    if not finite(p["sd"]) or p["sd"] < 0:
        msgs.append("Standar deviasi demand tidak tersedia (item butuh minimal 2 observasi, atau isi σd manual).")
    if not finite(pol["Q"]) or pol["Q"] <= 0:
        msgs.append("EOQ tidak dapat dihitung; periksa D, biaya pesan (S), dan biaya simpan (H) yang harus > 0.")
    if sim["source"] == SRC_BOOT and p["n"] < 2:
        msgs.append("Bootstrap membutuhkan minimal 2 observasi demand historis.")
    return msgs


# ---------------------------------------------------------------------------
# 5. Komponen tampilan
# ---------------------------------------------------------------------------
def store_result(key, sig, **data):
    st.session_state[key] = dict(sig=sig, **data)


def stale_notice(key, sig):
    stored = st.session_state.get(key)
    if stored is None:
        return None
    if stored["sig"] != sig:
        st.warning("Parameter atau pengaturan berubah sejak simulasi terakhir. Jalankan ulang agar hasil sesuai.")
    return stored


def tab_data(ctx):
    raw, d, m, notes = ctx["raw"], ctx["d"], ctx["m"], ctx["notes"]
    a, b, c = st.columns(3)
    a.metric("Baris", f"{len(raw):,}")
    b.metric("Kolom", f"{raw.shape[1]:,}")
    c.metric("Item (SKU)", f"{d['item'].nunique():,}")
    if "date" in d:
        st.caption(f"Rentang tanggal: {d['date'].min():%Y-%m-%d} sampai {d['date'].max():%Y-%m-%d}. "
                   f"Interval median antar tanggal unik: {fmt(ctx['gap'], 0)} hari.")
    for n in notes:
        st.warning(n)

    st.subheader("Pratinjau data")
    st.dataframe(raw.head(20))

    st.subheader("Tipe data dan missing values")
    info = pd.DataFrame({"Tipe": raw.dtypes.astype(str), "Terisi": raw.notna().sum(),
                         "Kosong": raw.isna().sum(), "% kosong": raw.isna().mean() * 100})
    st.dataframe(info.round(2))

    st.subheader("Statistik deskriptif (kolom numerik)")
    num = raw.select_dtypes("number")
    if num.empty:
        st.info("Tidak ada kolom numerik.")
    else:
        st.dataframe(num.describe().T.round(3))

    st.subheader("Pemetaan variabel yang dipakai")
    roles = [("Demand", m["demand"]), ("Tanggal / periode", m.get("date")),
             ("Item (SKU)", " + ".join(ctx["item_cols"]) if ctx["item_cols"] else None),
             ("Inventory level (observasi)", m.get("inv")), ("Unit terjual", m.get("sold")),
             ("Unit dipesan", m.get("ordered")), ("Unit cost", m.get("unit_cost")),
             ("Lead time", m.get("lead_time")), ("Ordering cost", m.get("order_cost")),
             ("Holding cost", m.get("hold_cost")), ("Stockout penalty", m.get("penalty"))]
    st.dataframe(pd.DataFrame({"Peran": [r[0] for r in roles],
                               "Kolom / sumber": [r[1] or "input manual / tidak dipakai" for r in roles]}).set_index("Peran"))


def tab_demand(ctx):
    d, g, item, stats, cfg = ctx["d"], ctx["g"], ctx["item"], ctx["stats"], ctx["cfg"]
    r = stats.loc[item]
    st.subheader(f"Demand: {item}")
    st.caption(f"Satuan: unit per periode ({ctx['freq_label'].lower()}). Hasil observasi dari data.")
    cols = st.columns(6)
    for col, (lab, val) in zip(cols, [("Rata-rata", r["mu"]), ("Median", r["med"]), ("Std", r["sd"]),
                                      ("Minimum", r["dmin"]), ("Maksimum", r["dmax"]), ("CV", r["cv"])]):
        col.metric(lab, fmt(val, 3 if lab == "CV" else 1))
    if r["n"] < 30:
        st.warning(f"Hanya {int(r['n'])} observasi; estimasi rata-rata dan σ kurang stabil.")

    if "date" in g:
        window = int(st.number_input("Jendela moving average (periode)", 1, 365, int(min(30, max(1, len(g) // 10))),
                                     key="ma_window"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["date"], y=g["demand"], mode="lines", name="Demand",
                                 line=dict(color=MUTED, width=1)))
        fig.add_trace(go.Scatter(x=g["date"], y=g["demand"].rolling(window, min_periods=1).mean(),
                                 mode="lines", name=f"Moving average {window}", line=dict(color=ACCENT, width=2)))
        st.plotly_chart(style(fig, "Demand terhadap waktu"))
        x = g["demand"].to_numpy(float)
        if len(x) > 3 and x.std() > 0:
            ctx["acf1"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            st.caption(f"Autokorelasi lag-1 demand: {ctx['acf1']:.3f} (dipakai sebagai pemeriksaan asumsi independensi).")
    else:
        st.info("Tidak ada kolom tanggal: grafik tren waktu tidak ditampilkan karena urutan baris tidak dijamin "
                "mewakili waktu.")

    fig = go.Figure(go.Histogram(x=g["demand"], marker_color=ACCENT, nbinsx=40))
    st.plotly_chart(style(fig, "Distribusi demand per periode", 300))

    st.subheader("Perbandingan antar item")
    if len(stats) > 1:
        tbl = stats[["n", "mu", "med", "sd", "dmin", "dmax", "cv"]].rename(columns={
            "n": "n", "mu": "Rata-rata", "med": "Median", "sd": "Std", "dmin": "Min", "dmax": "Maks", "cv": "CV"})
        st.dataframe(tbl.sort_values("Rata-rata", ascending=False).round(3))
        top = stats.sort_values("mu", ascending=False).head(25)
        fig = go.Figure(go.Bar(x=top.index, y=top["mu"], error_y=dict(type="data", array=top["sd"].fillna(0)),
                               marker_color=ACCENT))
        st.plotly_chart(style(fig, "Rata-rata demand ± 1 std (25 item teratas)"))
    else:
        st.caption("Hanya satu item pada data.")


def tab_policy(ctx):
    p, z, pol, cfg, stats = ctx["p"], ctx["z"], ctx["pol"], ctx["cfg"], ctx["stats"]
    st.subheader("Parameter masukan (item terpilih)")
    D = p["mu"] * p["ppy"]
    inputs = pd.DataFrame([
        ["d̄ (demand rata-rata)", p["mu"], "unit / periode", "data historis"],
        ["σd (std demand)", p["sd"], "unit / periode", p["src"]["sd"]],
        ["D = d̄ × periode per tahun", D, "unit / tahun", "dihitung"],
        ["S (biaya per pesanan)", p["S"], "biaya / pesanan", p["src"]["S"]],
        ["H (biaya simpan)", p["h"], "biaya / unit / tahun", p["src"]["h"]],
        ["L̄ (lead time)", p["L"], "periode", p["src"]["L"]],
        ["σL (std lead time)", p["sL"], "periode", p["src"]["L"]],
        ["Unit cost", p["c"], "biaya / unit", p["src"]["c"]],
    ], columns=["Parameter", "Nilai", "Satuan", "Sumber"]).set_index("Parameter")
    st.dataframe(inputs.assign(Nilai=inputs["Nilai"].map(lambda v: fmt(v, 4))))
    if any(v != "kolom data" and k in ("S", "h", "c", "pen") for k, v in p["src"].items()):
        st.caption("Biaya bersumber dari input manual di sidebar bukan dari data; hasil biaya hanya sebaik asumsi tersebut.")

    st.divider()
    st.subheader("EOQ")
    st.latex(r"EOQ = \sqrt{\dfrac{2\,D\,S}{H}}")
    if not finite(pol["Q"]):
        st.error("EOQ tidak dapat dihitung. D, S, dan H harus bernilai positif.")
    else:
        Q = pol["Q"]
        a, b, c, e = st.columns(4)
        a.metric("EOQ (unit)", fmt(Q, 1))
        b.metric("Pesanan / tahun", fmt(D / Q, 2))
        c.metric("Jarak antar pesanan (periode)", fmt(Q / p["mu"], 2))
        e.metric("Total biaya relevan EOQ / tahun", fmt(p["S"] * D / Q + p["h"] * Q / 2, 2))
        st.caption("Total biaya relevan EOQ = biaya pesan (S·D/Q) + biaya simpan (H·Q/2), tanpa safety stock "
                   "dan tanpa biaya pembelian. Pada Q = EOQ kedua komponen bernilai sama.")
        qs = np.linspace(0.2 * Q, 3 * Q, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qs, y=p["S"] * D / qs, name="Pemesanan", line=dict(color=MUTED, dash="dot")))
        fig.add_trace(go.Scatter(x=qs, y=p["h"] * qs / 2, name="Penyimpanan", line=dict(color=MUTED, dash="dash")))
        fig.add_trace(go.Scatter(x=qs, y=p["S"] * D / qs + p["h"] * qs / 2, name="Total", line=dict(color=ACCENT, width=2)))
        fig.add_vline(x=Q, line_color=INK, line_width=1)
        fig.update_xaxes(title="Q (unit)")
        fig.update_yaxes(title="Biaya / tahun")
        st.plotly_chart(style(fig, "Kurva biaya terhadap ukuran pesanan"))
        st.caption("Asumsi EOQ: demand konstan dan diketahui, penerimaan barang seketika (tanpa batas kapasitas), "
                   "tidak ada kekurangan, biaya konstan. Data nyata hampir selalu melanggar sebagian asumsi ini, "
                   "sehingga EOQ di sini adalah parameter kebijakan, bukan jaminan solusi optimal.")

    st.divider()
    st.subheader("Safety stock dan reorder point")
    constant_lt = p["sL"] <= 0
    if constant_lt:
        st.latex(r"SS = z\,\sigma_d \sqrt{L} \qquad ROP = \bar d\,L + SS")
        st.caption("Lead time konstan (σL = 0): suku variasi lead time bernilai nol.")
    else:
        st.latex(r"SS = z\,\sqrt{L\,\sigma_d^{2} + \bar d^{2}\,\sigma_L^{2}} \qquad ROP = \bar d\,\bar L + SS")
        st.caption("Demand dan lead time variabel. Rumus berlaku bila demand antar-periode saling independen, "
                   "lead time independen terhadap demand, dan keduanya dalam satuan periode yang sama.")
    a, b, c, e = st.columns(4)
    a.metric("Faktor z", fmt(z, 3))
    b.metric("Safety stock (unit)", fmt(pol["SS"], 1))
    c.metric("ROP (unit)", fmt(pol["ROP"], 1))
    e.metric("σ demand selama lead time", fmt(pol["sigma_lt"], 1))
    st.caption("ROP = rata-rata demand selama lead time + safety stock. Variabilitas hanya dihitung sekali di σ "
               "gabungan di atas; jangan menambah buffer lead time lagi secara terpisah.")
    if cfg["sd_mode"] == "manual":
        st.info("σd memakai input manual (mis. RMSE forecast per periode). Pastikan angka itu per periode, bukan "
                "per lead time, agar variabilitas tidak terhitung dua kali.")
    if "date" not in ctx["d"]:
        st.warning("Tanpa kolom tanggal, independensi antar-periode tidak dapat diperiksa.")
    elif finite(ctx.get("acf1")) and abs(ctx["acf1"]) > 0.3:
        st.warning(f"Autokorelasi lag-1 demand = {ctx['acf1']:.2f}. Asumsi independensi lemah; safety stock "
                   "analitis dapat keliru (terlalu kecil bila positif).")
    if p["n"] < 30:
        st.warning("Jumlah observasi item kecil; σd tidak stabil.")
    if finite(pol["Q"]):
        short = pol["sigma_lt"] * unit_loss(z)
        st.caption(f"Estimasi analitis (normal): kekurangan per siklus ≈ {fmt(short, 2)} unit, "
                   f"fill rate ≈ {fmt(max(0.0, 1 - short / pol['Q']), 4, True)}, "
                   f"cycle service level = target {fmt(cfg['sl'], 2, True)} bila asumsi terpenuhi.")

    st.divider()
    st.subheader("Seluruh item")
    if len(stats) > 1:
        st.dataframe(policy_table(stats, cfg, z).round(2))
        st.caption("Parameter biaya per item memakai kolom data bila tersedia, selain itu input manual yang sama untuk semua item.")
    else:
        st.caption("Hanya satu item pada data.")


def tab_observed(ctx):
    d, cfg, item = ctx["d"], ctx["cfg"], ctx["item"]
    st.subheader("Stockout, overstock, dan persediaan: hasil observasi")
    if "inv" not in d:
        st.info("Kolom inventory level tidak dipetakan, sehingga stockout, overstock, rata-rata persediaan, "
                "turnover, dan DIO observasi tidak dapat dihitung. Pilih kolom tersebut di sidebar bila ada.")
        return
    options = {"Demand − Inventory level (stok dianggap tersedia saat demand terjadi)": "stock"}
    if "sold" in d:
        options["Demand − Unit terjual"] = "sold"
    method = options[st.radio("Dasar unit tidak terpenuhi", list(options))]
    ctx["obs_method"] = method

    tbl = observed_table(d, cfg, method)
    if item not in tbl.index:
        st.warning("Item terpilih tidak memiliki baris dengan inventory level terisi.")
        return
    r = tbl.loc[item]
    a, b, c, e = st.columns(4)
    a.metric("Frekuensi stockout", fmt(r["stockout_freq"], 1, True))
    b.metric("Fill rate observasi", fmt(r["fill_rate"], 2, True))
    c.metric("Unit tidak terpenuhi", fmt(r["unmet"], 0))
    e.metric("Rata-rata persediaan", fmt(r["avg_inv"], 1))
    a, b, c, e = st.columns(4)
    a.metric("Frekuensi overstock", fmt(r["over_freq"], 1, True))
    b.metric("Rata-rata kelebihan (unit)", fmt(r["avg_excess"], 1))
    c.metric("Inventory turnover / tahun", fmt(r["turnover"], 2))
    e.metric("DIO (hari)", fmt(r["dio"], 1))

    with st.expander("Definisi dan asumsi metrik"):
        st.markdown(
            f"- **Stockout**: periode dengan unit tidak terpenuhi > 0. Unit tidak terpenuhi = max(0, basis), "
            f"dengan basis dipilih di atas. Data observasi tidak menunjukkan apakah kekurangan itu menjadi "
            f"*lost sales* atau *backorder*; simulasi memodelkan keduanya secara eksplisit.\n"
            f"- **Overstock**: periode dengan inventory level > {cfg['k_cover']:g} × rata-rata demand periode item "
            f"({fmt(cfg['k_cover'], 1)} periode cakupan, diatur di sidebar). Kelebihan = bagian stok di atas ambang; "
            f"stok di bawah ambang bukan overstock.\n"
            f"- **Fill rate** = 1 − unit tidak terpenuhi / total demand.\n"
            f"- **Turnover** = (unit terpenuhi per tahun) / rata-rata persediaan; **DIO** = 365 / turnover. "
            f"Dihitung dalam unit. Bila unit cost konstan per item, nilainya sama dengan versi berbasis HPP dan nilai "
            f"persediaan. Untuk gabungan banyak item dengan harga berbeda, versi unit tidak sama dengan versi nilai.\n"
            f"- Asumsi: satu baris = satu periode ({ctx['freq_label'].lower()}) dan inventory level adalah stok "
            f"yang tersedia untuk memenuhi demand pada baris yang sama."
        )

    if "date" in d:
        g = ctx["g"].dropna(subset=["inv"])
        if len(g):
            short = g[g["demand"] > g["inv"]] if method == "stock" else g[g["demand"] > g["sold"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g["date"], y=g["inv"], name="Inventory level", line=dict(color=ACCENT, width=1.5)))
            fig.add_trace(go.Scatter(x=g["date"], y=g["demand"], name="Demand", line=dict(color=MUTED, width=1)))
            fig.add_trace(go.Scatter(x=short["date"], y=short["demand"], mode="markers", name="Stockout",
                                     marker=dict(color=INK, size=5)))
            st.plotly_chart(style(fig, "Persediaan dan demand observasi"))

    st.subheader("Seluruh item")
    cols = ["n", "stockout_freq", "fill_rate", "unmet", "avg_inv", "over_freq", "avg_excess", "turnover", "dio"]
    st.dataframe(tbl[cols].round(3))


def tab_simulation(ctx):
    p, z, pol, cfg, sim = ctx["p"], ctx["z"], ctx["pol"], ctx["cfg"], ctx["sim"]
    st.subheader("Simulasi Monte Carlo: kebijakan (Q, R)")
    blockers = sim_blockers(p, pol, sim)
    if blockers:
        for m_ in blockers:
            st.error(m_)
        return

    manual = st.checkbox("Ubah parameter (Q, R) secara manual", value=False,
                         help="Default memakai EOQ dan ROP hasil perhitungan analitis.")
    over = (p["sd"] ** 2 + p["mu"] ** 2) / (2.0 * p["mu"])      # ekspektasi overshoot di bawah R per pesanan
    corr = False
    if manual:
        a, b = st.columns(2)
        Q = a.number_input("Q (unit)", min_value=0.1, value=float(max(round(pol["Q"], 1), 0.1)),
                           key=f"Q_{ctx['item']}_{pol['Q']:.4f}")
        R = b.number_input("R (unit)", min_value=0.0, value=float(max(round(pol["ROP"], 1), 0.0)),
                           key=f"R_{ctx['item']}_{pol['ROP']:.4f}")
    else:
        corr = st.checkbox("Koreksi overshoot diskret pada R", value=False,
                           help="Menaikkan R sebesar (σd² + d̄²) / (2d̄). Pendekatan, bukan hasil eksak.")
        Q, R = pol["Q"], pol["ROP"] + (over if corr else 0.0)
        if not corr and over > 0.5 * max(pol["SS"], 0.0):
            st.warning(
                f"Simulasi berjalan per periode: pesanan baru dipicu setelah stok melewati R, rata-rata sekitar "
                f"{fmt(over, 1)} unit di bawah R, sedangkan safety stock hanya {fmt(pol['SS'], 1)} unit. "
                "ROP analitis mengasumsikan review kontinu yang memicu tepat di R, sehingga CSL dan fill rate "
                "simulasi dapat jauh di bawah target. Centang koreksi overshoot untuk melihat efeknya."
            )
    ctx["Q"], ctx["R"], ctx["init_stock"], ctx["r_corr"] = Q, R, Q + R, corr

    L_sim = int(max(0, np.rint(p["L"])))
    st.caption(
        f"Kebijakan: pesan kelipatan Q = {fmt(Q, 1)} unit bila inventory position ≤ R = {fmt(R, 1)} unit. "
        f"Stok awal = R + Q = {fmt(Q + R, 1)} unit. Lead time simulasi = {L_sim} periode"
        + (f" (dibulatkan dari {fmt(p['L'], 2)})" if abs(p["L"] - L_sim) > 1e-9 and p["sL"] <= 0 else "")
        + (f", variabel dengan σL = {fmt(p['sL'], 2)} periode dan dibulatkan per pesanan" if p["sL"] > 0 else "")
        + f". Kekurangan diperlakukan sebagai {'backorder' if sim['backorder'] else 'lost sales'}. "
        f"Horizon {sim['horizon']:,} periode setelah warm-up {sim['warmup']:,}, {sim['n_rep']:,} replikasi, seed {sim['seed']}."
    )
    if p["L"] < 0.5:
        st.warning("Lead time kurang dari setengah periode data; simulasi membulatkannya menjadi 0 periode "
                   "(barang tiba di awal periode berikutnya).")

    sig = signature(cfg, ctx["item"], sim, Q, R, p)
    if st.button("Jalankan simulasi", key="run_sim"):
        res = simulate_policy(ctx, ("QR", Q, R))
        store_result("sim", sig, res=res, Q=Q, R=R, p=p)
    stored = stale_notice("sim", sig)
    if stored is None:
        st.info("Tekan 'Jalankan simulasi' untuk menghitung. Hasil disimpan sampai pengaturan berubah.")
        return

    res = stored["res"]
    means = sim_means(res)
    a, b, c, e = st.columns(4)
    a.metric("Total biaya relevan / tahun", fmt(means["total"], 0))
    b.metric("Fill rate (simulasi)", fmt(means["fill_rate"], 2, True))
    c.metric("Cycle service level (simulasi)", fmt(means["csl"], 2, True))
    e.metric("Rata-rata persediaan", fmt(means["avg_inv"], 1))
    st.dataframe(sim_summary_table(res))
    st.caption("Angka adalah rata-rata dan persentil 5-95 antar replikasi. Total biaya relevan = pemesanan + "
               "penyimpanan + stockout; biaya pembelian ditampilkan terpisah agar tidak terhitung ganda.")

    with st.expander("Definisi metrik dan asumsi simulasi"):
        st.markdown(
            "- **Cycle service level (CSL)**: peluang tidak terjadi stockout dalam satu siklus pemesanan. Dihitung "
            "sebagai 1 − (siklus dengan stockout / total siklus), siklus berakhir saat barang pesanan diterima.\n"
            "- **Fill rate**: porsi unit demand yang terpenuhi langsung dari stok = 1 − unit tidak terpenuhi / total demand. "
            "Satu siklus dengan kekurangan kecil menurunkan CSL penuh tetapi hampir tidak mengubah fill rate; "
            "keduanya tidak boleh dipertukarkan.\n"
            "- Keduanya adalah hasil **simulasi** dari model dan parameter di sidebar, bukan service level aktual perusahaan.\n"
            "- Biaya pemesanan dihitung per kejadian pemesanan; penyimpanan dari rata-rata persediaan (rata-rata stok awal "
            "dan akhir periode) × H; stockout = penalti × unit tidak terpenuhi (sekali per unit, juga pada backorder).\n"
            "- Demand per periode dibangkitkan i.i.d. (Normal dipotong di 0 atau bootstrap), sehingga tren, musiman, "
            "dan autokorelasi data asli tidak ikut dimodelkan. Lead time dibulatkan ke periode utuh.\n"
            "- Metrik dihitung setelah warm-up. Siklus yang sedang berjalan di akhir horizon tidak dihitung.\n"
            "- **Overshoot**: keputusan pesan dibuat di akhir periode, jadi inventory position biasanya sudah "
            "melewati R saat pesanan dibuat (rata-rata ≈ (σd² + d̄²) / (2d̄) unit). Rumus ROP analitis tidak "
            "memuatnya; pengaruhnya besar bila demand per periode jauh lebih besar daripada safety stock."
        )

    path = res["path"]
    x = np.arange(len(path["inv"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=path["inv"], name="Persediaan (akhir periode)", line=dict(color=ACCENT, width=1.5)))
    fig.add_hline(y=stored["R"], line_color=MUTED, line_dash="dash", annotation_text="R")
    placed = path["order"] > 0
    fig.add_trace(go.Scatter(x=x[placed], y=path["inv"][placed], mode="markers", name="Pesanan dibuat",
                             marker=dict(color=INK, size=6)))
    if path["warmup"] > 0:
        fig.add_vrect(x0=0, x1=path["warmup"], fillcolor=RULE, opacity=0.4, line_width=0)
    fig.update_xaxes(title="Periode simulasi")
    st.plotly_chart(style(fig, "Contoh satu replikasi (area abu-abu = warm-up)"))

    fig = go.Figure(go.Histogram(x=res["total"], marker_color=ACCENT, nbinsx=40))
    fig.update_xaxes(title="Total biaya relevan / tahun")
    st.plotly_chart(style(fig, "Sebaran total biaya antar replikasi", 300))


def tab_cost(ctx):
    d, p, z, pol, cfg, item = ctx["d"], ctx["p"], ctx["z"], ctx["pol"], ctx["cfg"], ctx["item"]
    st.subheader("Biaya dan kinerja persediaan")
    st.caption("Tiga sumber dibedakan: observasi dari data (proksi), perhitungan analitis (aproksimasi normal), "
               "dan rata-rata hasil simulasi. Ketiganya memakai parameter biaya yang sama.")
    if not finite(pol["Q"]):
        st.error("Kebijakan analitis tidak dapat dihitung; lihat tab EOQ.")
        return

    years_obs = np.nan
    obs = dict.fromkeys(["ordering", "holding", "stockout", "purchase", "total", "fill_rate", "csl",
                         "avg_inv", "turnover", "dio"], np.nan)
    if "inv" in d:
        tbl = observed_table(d, cfg, ctx.get("obs_method", "stock"))
        if item in tbl.index:
            r = tbl.loc[item]
            years_obs = r["n"] / p["ppy"]
            obs["holding"] = p["h"] * r["avg_inv"]
            obs["stockout"] = p["pen"] * r["unmet"] / years_obs
            if "n_orders" in tbl:
                obs["ordering"] = p["S"] * r["n_orders"] / years_obs
                obs["purchase"] = p["c"] * r["ordered_units"] / years_obs
            if all(finite(obs[k]) for k in ("ordering", "holding", "stockout")):
                obs["total"] = obs["ordering"] + obs["holding"] + obs["stockout"]
            obs.update(fill_rate=r["fill_rate"], avg_inv=r["avg_inv"], turnover=r["turnover"], dio=r["dio"])
    ana = analytic_costs(p, pol, z)
    ana["csl"] = cfg["sl"]

    sim_vals = dict.fromkeys(obs, np.nan)
    sim_store = st.session_state.get("sim")
    if sim_store is not None:
        sim_vals = sim_means(sim_store["res"])
        if sim_store["sig"] != signature(cfg, item, ctx["sim"], sim_store["Q"], sim_store["R"], p):
            st.warning("Kolom Simulasi berasal dari pengaturan sebelumnya; jalankan ulang di tab Simulasi.")
    else:
        st.info("Kolom Simulasi kosong: jalankan simulasi di tab Simulasi.")

    rows = [("ordering", "Biaya pemesanan / tahun", 2, False), ("holding", "Biaya penyimpanan / tahun", 2, False),
            ("stockout", "Biaya stockout / tahun", 2, False), ("total", "Total biaya relevan / tahun", 2, False),
            ("purchase", "Biaya pembelian / tahun (di luar total)", 2, False),
            ("fill_rate", "Fill rate", 2, True), ("csl", "Cycle service level", 2, True),
            ("avg_inv", "Rata-rata persediaan (unit)", 1, False), ("turnover", "Inventory turnover / tahun", 2, False),
            ("dio", "DIO (hari)", 1, False)]
    table = pd.DataFrame({
        "Metrik": [r[1] for r in rows],
        "Observasi (proksi)": [fmt(obs.get(k), nd, pc) for k, _, nd, pc in rows],
        "Analitis (Q = EOQ, R = ROP)": [fmt(ana.get(k), nd, pc) for k, _, nd, pc in rows],
        "Simulasi (rata-rata)": [fmt(sim_vals.get(k), nd, pc) for k, _, nd, pc in rows],
    }).set_index("Metrik")
    st.dataframe(table)

    with st.expander("Cara membaca dan keterbatasan"):
        st.markdown(
            "- **Observasi**: penyimpanan = H × rata-rata inventory level; stockout = penalti × unit tidak terpenuhi per tahun; "
            "pemesanan dan pembelian hanya dihitung bila kolom unit dipesan dipetakan (setiap baris dengan nilai > 0 dianggap "
            "satu pesanan). Total observasi kosong bila salah satu komponen tidak tersedia. Ini proksi bukan biaya akuntansi.\n"
            "- **Analitis**: pemesanan = S·D/Q; penyimpanan = H·(Q/2 + SS); stockout = penalti × (D/Q) × σ_LT × G(z); "
            "fill rate ≈ 1 − σ_LT·G(z)/Q. Aproksimasi, valid saat demand selama lead time mendekati normal.\n"
            "- **Simulasi**: lihat tab Simulasi. CSL dan fill rate simulasi bukan service level aktual.\n"
            "- CSL observasi tidak dihitung karena siklus pemesanan tidak dapat diidentifikasi dari data ini.\n"
            "- Selisih analitis vs simulasi pada (Q, R) wajar: simulasi bekerja per periode (ada overshoot di bawah R, "
            "lihat tab Simulasi), sedangkan rumus analitis mengasumsikan review kontinu.\n"
            "- Turnover dan DIO berbasis unit; sama dengan versi berbasis nilai bila unit cost konstan untuk item yang sama."
        )

    fig = go.Figure()
    for name, vals, color in (("Observasi", obs, MUTED), ("Analitis", ana, RULE), ("Simulasi", sim_vals, ACCENT)):
        if any(finite(vals.get(k)) for k in ("ordering", "holding", "stockout")):
            fig.add_trace(go.Bar(name=name, x=["Pemesanan", "Penyimpanan", "Stockout"],
                                 y=[vals.get(k) if finite(vals.get(k)) else None for k in ("ordering", "holding", "stockout")],
                                 marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Biaya / tahun")
    st.plotly_chart(style(fig, "Komponen biaya per sumber"))


def sensitivity_rows(ctx, param, mults, mode):
    """Satu baris hasil per pengali. mode: 'fixed' (Q, R dasar) atau 'recompute' (EOQ/ROP dihitung ulang)."""
    p, z = ctx["p"], ctx["z"]
    rows = []
    for mult in mults:
        m = {param: mult}
        ps = apply_multipliers(p, m)
        pol_s = analytic_policy(ps, z)
        if mode == "fixed":
            Q, R = ctx["Q"], ctx["R"]
        else:
            Q, R = pol_s["Q"], pol_s["ROP"]
            if ctx.get("r_corr") and ps["mu"] > 0:
                R += (ps["sd"] ** 2 + ps["mu"] ** 2) / (2.0 * ps["mu"])
        if not (finite(Q) and Q > 0):
            continue
        res = simulate_policy(ctx, ("QR", Q, R), mult=(m.get("mean", 1.0), m.get("std", 1.0), m.get("lt", 1.0)),
                              cost=(ps["S"], ps["h"], ps["pen"], p["c"]))
        v = sim_means(res)
        rows.append({"param": param, "Pengali": mult, "Q": Q, "R": R, **v})
    return pd.DataFrame(rows)


SENS_PARAMS = {"Rata-rata demand": "mean", "Variasi demand (σd)": "std", "Lead time": "lt",
               "Holding cost": "h", "Ordering cost": "S", "Stockout penalty": "pen"}


def tab_sensitivity(ctx):
    p, pol, cfg, sim = ctx["p"], ctx["pol"], ctx["cfg"], ctx["sim"]
    st.subheader("Analisis sensitivitas")
    blockers = sim_blockers(p, pol, sim)
    if blockers:
        st.error(blockers[0])
        return
    st.caption("Satu parameter diubah dengan pengali sementara yang lain tetap. Semua skenario memakai seed, horizon, "
               "warm-up, dan stok awal yang sama sehingga perbedaan hasil berasal dari parameter, bukan dari angka acak.")
    view = st.radio("Tampilan", ["Satu parameter (rentang)", "Semua parameter (batas bawah dan atas)"], horizontal=True)
    lo, hi = st.slider("Rentang pengali", 0.25, 3.0, (0.5, 1.5), 0.05)
    mode_label = st.radio("Perlakuan kebijakan", ["Q dan R tetap (uji ketahanan)", "EOQ dan ROP dihitung ulang tiap skenario"],
                          horizontal=True)
    mode = "fixed" if mode_label.startswith("Q dan R tetap") else "recompute"
    if view.startswith("Satu"):
        name = st.selectbox("Parameter", list(SENS_PARAMS))
        n_pts = st.slider("Jumlah titik", 3, 9, 5)
        job = ("one", SENS_PARAMS[name], tuple(np.linspace(lo, hi, n_pts)))
    else:
        job = ("all", None, (lo, hi))
    sig = signature(cfg, ctx["item"], sim, ctx["Q"], ctx["R"], p, job, mode)
    if st.button("Jalankan analisis sensitivitas", key="run_sens"):
        with st.spinner("Menjalankan skenario..."):
            if job[0] == "one":
                frame = sensitivity_rows(ctx, job[1], job[2], mode)
            else:
                parts = [sensitivity_rows(ctx, code, job[2], mode).assign(Parameter=lab)
                         for lab, code in SENS_PARAMS.items()]
                frame = pd.concat(parts, ignore_index=True)
            base = simulate_policy(ctx, ("QR", ctx["Q"], ctx["R"]))
        store_result("sens", sig, frame=frame, job=job, base=sim_means(base))
    stored = stale_notice("sens", sig)
    if stored is None:
        st.info("Tekan tombol di atas untuk menjalankan skenario.")
        return
    frame, job, base = stored["frame"], stored["job"], stored["base"]
    if frame.empty:
        st.error("Tidak ada skenario yang valid (EOQ tidak dapat dihitung pada rentang ini).")
        return

    if job[0] == "one":
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Total biaya relevan / tahun", "Proporsi periode stockout",
                                                           "Fill rate dan CSL", "Rata-rata persediaan"))
        x = frame["Pengali"]
        fig.add_trace(go.Scatter(x=x, y=frame["total"], mode="lines+markers", line=dict(color=ACCENT), showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=frame["stockout_freq"], mode="lines+markers", line=dict(color=ACCENT), showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=x, y=frame["fill_rate"], name="Fill rate", mode="lines+markers", line=dict(color=ACCENT)), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=frame["csl"], name="CSL", mode="lines+markers", line=dict(color=MUTED, dash="dash")), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=frame["avg_inv"], mode="lines+markers", line=dict(color=ACCENT), showlegend=False), 2, 2)
        fig.update_xaxes(title="Pengali parameter")
        st.plotly_chart(style(fig, None, 560))
        show = frame.rename(columns={"total": "Total biaya", "ordering": "Pemesanan", "holding": "Penyimpanan",
                                     "stockout": "Stockout", "fill_rate": "Fill rate", "csl": "CSL",
                                     "stockout_freq": "Prop. periode stockout", "avg_inv": "Rata-rata persediaan"})
        st.dataframe(show[["Pengali", "Q", "R", "Total biaya", "Pemesanan", "Penyimpanan", "Stockout", "Fill rate",
                           "CSL", "Prop. periode stockout", "Rata-rata persediaan"]].set_index("Pengali").round(3))
    else:
        lo_v, hi_v = job[2]
        wide = frame.pivot_table(index="Parameter", columns="Pengali", values=["total", "fill_rate", "stockout_freq", "avg_inv"])
        out = pd.DataFrame(index=wide.index)
        for tag, val in (("batas bawah", lo_v), ("batas atas", hi_v)):
            col = min(wide["total"].columns, key=lambda c_: abs(c_ - val))
            out[f"Δ total biaya, {tag} (%)"] = (wide["total"][col] / base["total"] - 1) * 100
            out[f"Fill rate, {tag}"] = wide["fill_rate"][col]
            out[f"Prop. stockout, {tag}"] = wide["stockout_freq"][col]
            out[f"Rata-rata persediaan, {tag}"] = wide["avg_inv"][col]
        out = out.loc[(out.filter(like="Δ").abs().max(axis=1)).sort_values(ascending=False).index]
        st.dataframe(out.round(3))
        fig = go.Figure()
        fig.add_trace(go.Bar(y=out.index, x=out.iloc[:, 0], orientation="h", name=f"Pengali {lo_v:g}", marker_color=MUTED))
        hi_col = [c_ for c_ in out.columns if c_.startswith("Δ total biaya, batas atas")][0]
        fig.add_trace(go.Bar(y=out.index, x=out[hi_col], orientation="h", name=f"Pengali {hi_v:g}", marker_color=ACCENT))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Perubahan total biaya relevan terhadap kondisi dasar (%)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, None, 380))
        st.caption(f"Kondisi dasar: total biaya relevan {fmt(base['total'], 0)} per tahun, fill rate {fmt(base['fill_rate'], 2, True)}.")
    st.caption("Pengali variasi demand mengubah deviasi di sekitar rata-rata; pengali rata-rata demand mengubah seluruh "
               "distribusi. Lead time simulasi dibulatkan ke periode utuh sehingga efeknya berbentuk tangga.")


def tab_compare(ctx):
    p, z, pol, cfg, sim = ctx["p"], ctx["z"], ctx["pol"], ctx["cfg"], ctx["sim"]
    st.subheader("Perbandingan kebijakan: (Q, R) dan (P, S)")
    blockers = sim_blockers(p, pol, sim)
    if blockers:
        st.error(blockers[0])
        return
    Q, R = ctx["Q"], ctx["R"]
    st.markdown("**(Q, R)** continuous review: pesan Q bila inventory position ≤ R.  \n"
                "**(P, S)** periodic review: tiap P periode pesan sebesar S − inventory position.")
    P0 = int(min(3650, max(1, round(pol["Q"] / p["mu"])))) if p["mu"] > 0 else 1
    P = int(st.number_input("Interval review P (periode)", 1, 3650, P0, key=f"P_{ctx['item']}_{P0}",
                            help="Default membulatkan EOQ / demand rata-rata per periode."))
    ps = periodic_policy(p, z, P)
    S_level = ps["S_level"]
    if st.checkbox("Ubah order-up-to level S manual", value=False):
        S_level = st.number_input("S (unit)", min_value=0.0, value=float(round(ps["S_level"], 1)),
                                  key=f"Slev_{ctx['item']}_{P}_{ps['S_level']:.3f}")
    st.caption(f"(P, S): P = {P} periode, S = {fmt(S_level, 1)} unit. S = d̄(P + L) + z·√((P + L)σd² + d̄²σL²). "
               f"Kedua kebijakan memakai stok awal yang sama ({fmt(ctx['init_stock'], 1)} unit), seed, horizon, dan angka acak yang sama.")

    sig = signature(cfg, ctx["item"], sim, Q, R, P, S_level, p)
    if st.button("Bandingkan kebijakan", key="run_cmp"):
        res_qr = simulate_policy(ctx, ("QR", Q, R))
        res_ps = simulate_policy(ctx, ("PS", P, S_level))
        store_result("cmp", sig, qr=res_qr, ps=res_ps)
    stored = stale_notice("cmp", sig)
    if stored is None:
        st.info("Tekan 'Bandingkan kebijakan' untuk menjalankan simulasi kedua kebijakan.")
        return

    qr, psr = stored["qr"], stored["ps"]
    mq, mp = sim_means(qr), sim_means(psr)
    keys = [("total", "Total biaya relevan / tahun", 2, False), ("ordering", "Biaya pemesanan", 2, False),
            ("holding", "Biaya penyimpanan", 2, False), ("stockout", "Biaya stockout", 2, False),
            ("fill_rate", "Fill rate", 2, True), ("csl", "Cycle service level", 2, True),
            ("stockout_freq", "Proporsi periode stockout", 2, True), ("avg_inv", "Rata-rata persediaan (unit)", 1, False),
            ("orders_per_year", "Pesanan / tahun", 2, False)]
    table = pd.DataFrame({
        "Metrik": [k[1] for k in keys],
        "(Q, R)": [fmt(mq[k], nd, pc) for k, _, nd, pc in keys],
        "(P, S)": [fmt(mp[k], nd, pc) for k, _, nd, pc in keys],
    }).set_index("Metrik")
    st.dataframe(table)

    n = len(qr["total"])
    rows = []
    for key, label, nd, pc in [("total", "Total biaya relevan / tahun", 2, False), ("fill_rate", "Fill rate", 4, True),
                               ("avg_inv", "Rata-rata persediaan (unit)", 2, False)]:
        diff = psr[key] - qr[key]
        se = np.nanstd(diff, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        mean = np.nanmean(diff)
        rows.append({"Selisih (P, S) − (Q, R)": label, "Rata-rata": fmt(mean, nd, pc),
                     "Interval 95%": f"{fmt(mean - 1.96 * se, nd, pc)} s/d {fmt(mean + 1.96 * se, nd, pc)}"})
    st.dataframe(pd.DataFrame(rows).set_index("Selisih (P, S) − (Q, R)"))
    st.caption("Selisih dihitung berpasangan per replikasi (angka acak sama). Interval yang melewati nol berarti "
               "perbedaan tidak jelas pada jumlah replikasi ini.")

    fig = go.Figure()
    for name, m_, color in (("(Q, R)", mq, ACCENT), ("(P, S)", mp, MUTED)):
        fig.add_trace(go.Bar(name=name, x=["Pemesanan", "Penyimpanan", "Stockout"],
                             y=[m_["ordering"], m_["holding"], m_["stockout"]], marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Biaya / tahun")
    st.plotly_chart(style(fig, "Komponen biaya"))

    fig = go.Figure()
    for name, m_, color in (("(Q, R)", mq, ACCENT), ("(P, S)", mp, MUTED)):
        fig.add_trace(go.Scatter(x=[m_["avg_inv"]], y=[m_["fill_rate"]], mode="markers+text", text=[name],
                                 textposition="top center", marker=dict(size=14, color=color), name=name))
    fig.update_xaxes(title="Rata-rata persediaan (unit)")
    fig.update_yaxes(title="Fill rate")
    st.plotly_chart(style(fig, "Trade-off persediaan dan tingkat layanan", 320))
    st.info("Tidak ada satu kebijakan yang dinyatakan terbaik. Kebijakan dengan biaya lebih rendah dapat memiliki "
            "risiko stockout atau persediaan yang berbeda; pilih berdasarkan prioritas biaya, layanan, dan modal kerja. "
            "Hasil berlaku untuk asumsi demand i.i.d. dan parameter biaya di sidebar.")


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Inventory Control Dashboard", layout="wide")
    st.title("Inventory Control Dashboard")
    st.caption("EOQ, safety stock, reorder point, simulasi Monte Carlo, dan perbandingan kebijakan persediaan.")

    # ---- Sidebar bagian 1: dataset dan pemetaan variabel
    with st.sidebar:
        st.header("Dataset")
        upload = st.file_uploader("File CSV", type=["csv"])
        sep = st.selectbox("Pemisah kolom", [",", ";", "\t", "|"], format_func=lambda s: "Tab" if s == "\t" else s)

    if upload is None:
        st.info("Unggah file CSV di sidebar untuk memulai. Minimal dibutuhkan satu kolom demand; kolom lain bersifat "
                "opsional dan menentukan analisis mana yang dapat dihitung (lihat daftar kolom di dokumentasi).")
        st.stop()
    try:
        raw = load_csv(upload.getvalue(), sep)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        st.error(f"CSV tidak dapat dibaca: {e}. Coba ubah pemisah kolom.")
        st.stop()
    if raw.shape[1] < 2:
        st.error("CSV hanya terbaca sebagai satu kolom. Kemungkinan pemisah kolom salah; ubah di sidebar.")
        st.stop()
    cols = list(raw.columns)

    def pick(label, role, help=None):
        choice = st.selectbox(label, [NONE] + cols, index=guess_index(cols, HINTS[role]), help=help)
        return None if choice == NONE else choice

    with st.sidebar:
        st.header("Variabel")
        m = {"demand": pick("Demand (wajib)", "demand")}
        m["date"] = pick("Tanggal / periode", "date")
        item_cols = st.multiselect("Kolom identitas item (SKU)", cols,
                                   default=[c for c in cols if c.lower().strip() in ITEM_HINTS],
                                   help="Beberapa kolom digabung menjadi satu item, mis. Store ID + Product ID.")
        m["inv"] = pick("Inventory level (observasi)", "inv")
        with st.expander("Kolom opsional lain"):
            m["sold"] = pick("Unit terjual", "sold")
            m["ordered"] = pick("Unit dipesan", "ordered")
            m["unit_cost"] = pick("Unit cost", "unit_cost", "Biaya perolehan per unit, bukan harga jual.")
            m["lead_time"] = pick("Lead time", "lead_time")
            m["order_cost"] = pick("Ordering cost per pesanan", "order_cost")
            m["hold_cost"] = pick("Holding cost per unit per tahun", "hold_cost")
            m["penalty"] = pick("Stockout penalty per unit", "penalty")
        missing_rule = st.selectbox("Demand kosong", ["Buang baris", "Isi dengan 0"])
        slot_item = st.container()

    if m["demand"] is None:
        st.info("Pilih kolom demand di sidebar.")
        st.stop()
    if m["demand"] in item_cols:
        st.error("Kolom demand tidak boleh sekaligus menjadi kolom identitas item.")
        st.stop()
    try:
        d, notes = prepare_data(raw, {k: v for k, v in m.items()}, tuple(item_cols), missing_rule)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # ---- Sidebar bagian 2: frekuensi, parameter, simulasi
    gap = infer_gap_days(d["date"]) if "date" in d else None
    stats = item_stats_table(d)
    with slot_item:
        if len(stats) > 1:
            item = st.selectbox("Item untuk analisis detail", list(stats.index))
        else:
            item = stats.index[0]

    with st.sidebar:
        st.header("Frekuensi data")
        freq_label = st.selectbox("Satu baris mewakili", list(FREQUENCIES), index=freq_default(gap))
        if FREQUENCIES[freq_label] is None:
            ppy = float(st.number_input("Jumlah periode per tahun", 1.0, 3650.0, 365.0))
        else:
            ppy = FREQUENCIES[freq_label]
        if gap is None:
            st.caption("Tanpa tanggal, frekuensi tidak dapat diverifikasi dari data. Pastikan pilihan ini benar.")
        else:
            st.caption(f"Interval median antar tanggal: {gap:g} hari.")

        st.header("Parameter")
        sl = st.number_input("Target service level (CSL)", 0.50, 0.999, 0.95, 0.005, format="%.3f")
        sd_mode = st.radio("Sumber σ demand", ["Data historis", "Input manual (mis. RMSE forecast)"])
        sd_manual = st.number_input("σ demand per periode (manual)", 0.0, 1e9, 10.0) if sd_mode.startswith("Input") else 0.0
        lt_unit = st.radio("Satuan lead time", ["Hari", "Periode data"], horizontal=True)
        lt_mean = st.number_input("Lead time rata-rata (input manual)", 0.0, 1e4, 3.0,
                                  help="Dipakai bila tidak ada kolom lead time.")
        lt_sd = st.number_input("σ lead time (input manual)", 0.0, 1e4, 0.0)
        st.caption("Nilai biaya di bawah adalah asumsi awal, bukan hasil dari data. Ganti sesuai konteks. "
                   "Kolom data, bila dipetakan, diutamakan.")
        c_manual = st.number_input("Unit cost", 0.0, 1e9, 10.0)
        S_manual = st.number_input("Ordering cost per pesanan", 0.0, 1e9, 50.0)
        h_mode_label = st.radio("Holding cost", ["Persentase dari unit cost per tahun", "Nilai tetap per unit per tahun"])
        h_rate = st.number_input("Persentase holding cost per tahun", 0.0, 5.0, 0.20, 0.01) if h_mode_label.startswith("Persen") else 0.0
        h_fixed = st.number_input("Holding cost per unit per tahun", 0.0, 1e9, 2.0) if h_mode_label.startswith("Nilai") else 0.0
        pen_manual = st.number_input("Stockout penalty per unit", 0.0, 1e9, 3.0)
        k_cover = st.number_input("Ambang overstock (k × demand rata-rata periode)", 0.5, 365.0, 5.0,
                                  help="Overstock = persediaan di atas k periode cakupan demand.")

        st.header("Simulasi")
        n_rep = int(st.number_input("Jumlah replikasi", 50, 5000, 300, 50))
        horizon = int(st.number_input("Horizon (periode)", 10, 5000, int(min(ppy, 1000)) if ppy >= 10 else 100))
        warmup = int(st.number_input("Warm-up (periode, tidak dihitung)", 0, 2000, 30))
        seed = int(st.number_input("Random seed", 0, 2 ** 31 - 1, 42))
        source = st.radio("Distribusi demand simulasi", [SRC_NORMAL, SRC_BOOT])
        backorder = st.radio("Perlakuan kekurangan", ["Lost sales", "Backorder"], horizontal=True) == "Backorder"

    cfg = dict(ppy=ppy, sl=sl, sd_mode="manual" if sd_mode.startswith("Input") else "data", sd_manual=sd_manual,
               lt_unit=lt_unit, lt_mean=lt_mean, lt_sd=lt_sd, c_manual=c_manual, S_manual=S_manual,
               h_mode="rate" if h_mode_label.startswith("Persen") else "fixed", h_rate=h_rate, h_fixed=h_fixed,
               pen_manual=pen_manual, k_cover=k_cover)
    sim = dict(n_rep=n_rep, horizon=horizon, warmup=warmup, seed=seed, source=source, backorder=backorder)

    g = d[d["item"] == item]
    z = z_of(sl)
    p = params_from_stats(stats.loc[item], cfg)
    pol = analytic_policy(p, z)
    ctx = dict(raw=raw, d=d, m=m, notes=notes, item_cols=item_cols, gap=gap, stats=stats, item=item, g=g,
               cfg=cfg, sim=sim, p=p, z=z, pol=pol, freq_label=freq_label, r_corr=False,
               hist=g["demand"].to_numpy(float), Q=pol["Q"], R=pol["ROP"],
               init_stock=(pol["Q"] + pol["ROP"]) if finite(pol["Q"]) else 0.0)

    tabs = st.tabs(["Data", "Demand", "EOQ, Safety Stock, ROP", "Observasi", "Simulasi",
                    "Biaya dan Kinerja", "Sensitivitas", "Perbandingan Kebijakan"])
    with tabs[0]:
        tab_data(ctx)
    with tabs[1]:
        tab_demand(ctx)
    with tabs[2]:
        tab_policy(ctx)
    with tabs[3]:
        tab_observed(ctx)
    with tabs[4]:
        tab_simulation(ctx)
    with tabs[5]:
        tab_cost(ctx)
    with tabs[6]:
        tab_sensitivity(ctx)
    with tabs[7]:
        tab_compare(ctx)


if __name__ == "__main__":
    main()
