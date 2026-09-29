"""
Gold Trading Analyst - Streamlit dashboard
Run:  streamlit run app.py
"""
import html as htmlmod
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Gold Trading Analyst", page_icon="🥇", layout="wide")

# Key comes from .streamlit/secrets.toml (ALPHAVANTAGE_API_KEY = "...") or env var.
try:
    API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
except Exception:
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

AV = "https://www.alphavantage.co/query"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

AV_NEWS_FEEDS = {
    "Monetary Policy": {"topics": "economy_monetary", "sort": "LATEST", "limit": 50},
    "Gold": {"tickers": "Gold", "sort": "LATEST", "limit": 50},
    "Macro & Markets": {
        "topics": "economy_monetary,economy_macro,financial_markets,energy_transportation",
        "sort": "LATEST",
        "limit": 100,
    },
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem;}
.hero {background: linear-gradient(135deg,#1a1405 0%,#3a2b06 60%,#7a5a0c 100%);
       padding: 1.4rem 1.8rem; border-radius: 16px; color: #fff; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size: 2rem; color:#ffd24d;}
.hero p {margin:.2rem 0 0 0; color:#e8dcc0;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:.9rem 1.1rem;
       margin-bottom:.7rem; background: rgba(255,210,77,.04);}
.card h4 {margin:0 0 .3rem 0; font-size:1.02rem;}
.card a {text-decoration:none;}
.meta {font-size:.78rem; opacity:.7;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
        font-weight:600; margin-right:6px;}
.bull {background:#0f5132; color:#d1f7e2;}
.bear {background:#842029; color:#f8d7da;}
.neut {background:#41464b; color:#e2e3e5;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Alpha Vantage helpers
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def av_get(params: dict) -> dict:
    if not API_KEY:
        return {"error": "Alpha Vantage API key missing (set ALPHAVANTAGE_API_KEY)."}
    try:
        r = requests.get(AV, params={**params, "apikey": API_KEY}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    # Alpha Vantage returns 200 with a note/information key when rate limited
    for k in ("Note", "Information", "Error Message"):
        if k in data:
            return {"error": data[k]}
    return data


def label_class(label: str) -> str:
    l = (label or "").lower()
    if "bullish" in l:
        return "bull"
    if "bearish" in l:
        return "bear"
    return "neut"


def fmt_av_time(ts: str) -> str:
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%S").strftime("%d %b %Y, %H:%M")
    except Exception:  # noqa: BLE001
        return ts or ""


def render_av_news(data: dict, max_items: int = 20):
    if "error" in data:
        st.warning(data["error"])
        return
    feed = data.get("feed", [])[:max_items]
    if not feed:
        st.info("No articles returned.")
        return
    scores = [a.get("overall_sentiment_score", 0) for a in feed]
    avg = sum(scores) / len(scores)
    c1, c2, c3 = st.columns(3)
    c1.metric("Articles", len(feed))
    c2.metric("Avg sentiment", f"{avg:+.3f}")
    c3.metric(
        "Bullish / Bearish",
        f"{sum(s > 0.15 for s in scores)} / {sum(s < -0.15 for s in scores)}",
    )
    for a in feed:
        cls = label_class(a.get("overall_sentiment_label"))
        summary = (a.get("summary") or "")[:280]
        st.markdown(
            f"""<div class="card">
<span class="badge {cls}">{a.get('overall_sentiment_label','')}</span>
<span class="meta">{a.get('source','')} · {fmt_av_time(a.get('time_published',''))}</span>
<h4><a href="{a.get('url','#')}" target="_blank">{a.get('title','')}</a></h4>
<div class="meta">{summary}…</div></div>""",
            unsafe_allow_html=True,
        )


def av_series_df(data: dict) -> pd.DataFrame | None:
    if "error" in data or "data" not in data:
        return None
    df = pd.DataFrame(data["data"])
    if df.empty:
        return None
    # Different endpoints name columns differently (value / price / close ...)
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[k] for k in ("date", "timestamp", "time") if k in cols), None)
    if date_col is None:
        return None
    val_col = next(
        (cols[k] for k in ("value", "price", "close", "nominal", "usd", "rate") if k in cols),
        None,
    )
    if val_col is None:  # fall back to the first non-date column
        others = [c for c in df.columns if c != date_col]
        if not others:
            return None
        val_col = others[0]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[val_col], errors="coerce"),
        }
    )
    out = out.dropna().sort_values("date")
    return out if not out.empty else None


def line_chart(df: pd.DataFrame, title: str, color: str, days: int = 180, key: str | None = None):
    df = df.tail(days)
    fig = go.Figure(go.Scatter(x=df["date"], y=df["value"], mode="lines",
                               line=dict(color=color, width=2), fill="tozeroy",
                               fillcolor="rgba(255,210,77,0.06)"))
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      template="plotly_dark", yaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True, key=key or f"chart_{title}")


def delta_metric(col, label, df, suffix=""):
    if df is None or len(df) < 2:
        col.metric(label, "n/a")
        return
    last, prev = df["value"].iloc[-1], df["value"].iloc[-2]
    col.metric(label, f"{last:,.2f}{suffix}", f"{last - prev:+.2f}")


# ----------------------------------------------------------------------------
# Scraping helpers (FXStreet / Bloomberg)
# ----------------------------------------------------------------------------
def fetch_html(url: str) -> tuple[str | None, str | None]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return r.text, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def scrape_fxstreet_links(url: str, limit: int) -> tuple[list[dict], str | None]:
    """Collect article links (/news/<slug>) from an FXStreet listing page."""
    html, err = fetch_html(url)
    if err:
        return [], err
    soup = BeautifulSoup(html, "html.parser")
    seen, items = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.match(r"^(https://www\.fxstreet\.com)?/news/[a-z0-9\-]{15,}", href):
            continue
        full = href if href.startswith("http") else f"https://www.fxstreet.com{href}"
        title = a.get_text(" ", strip=True)
        if full in seen or len(title) < 20:
            continue
        seen.add(full)
        items.append({"title": title, "url": full})
        if len(items) >= limit:
            break
    if not items:
        return [], "No articles parsed (page layout may have changed or is JS-rendered)."
    return items, None


@st.cache_data(ttl=300, show_spinner=False)
def fx_main(limit=8):
    return scrape_fxstreet_links("https://www.fxstreet.com/", limit)


@st.cache_data(ttl=300, show_spinner=False)
def fx_commodities(limit=3):
    return scrape_fxstreet_links("https://www.fxstreet.com/news?Tags=Commodities", limit)


@st.cache_data(ttl=300, show_spinner=False)
def fx_generic(limit=5):
    return scrape_fxstreet_links("https://www.fxstreet.com/news", limit)


@st.cache_data(ttl=300, show_spinner=False)
def bloomberg_rates():
    """Bloomberg is heavily bot-protected; this often returns 403. We try, and fail gracefully."""
    html, err = fetch_html("https://www.bloomberg.com/markets/rates-bonds")
    if err:
        return None, err
    try:
        tables = pd.read_html(html)
        if tables:
            return tables[0], None
    except Exception:  # noqa: BLE001
        pass
    return None, "Page loaded but no rate table found (content is JS-rendered)."


def render_links(items, err, empty_msg="Nothing to show."):
    if err:
        st.warning(f"Could not load: {err}")
        return
    if not items:
        st.info(empty_msg)
        return
    for it in items:
        st.markdown(
            f"""<div class="card"><h4><a href="{it['url']}" target="_blank">{it['title']}</a></h4>
<div class="meta">FXStreet</div></div>""",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Google News RSS
# ----------------------------------------------------------------------------
# Edit these queries to steer the news flow. "when:2d" (added below) limits to last 48h.
GOOGLE_QUERIES = {
    "XAU/USD": "XAU/USD",
    "Gold forecast": "gold price forecast next move",
    "Fed & rates": "gold Fed rate cut hike expectations",
    "Dollar & yields": "gold US dollar Treasury yields DXY",
    "Geopolitics": "gold safe haven geopolitical tensions",
}

BULL_WORDS = ["surge", "rally", "rallies", "rise", "rises", "jump", "record", "soar", "climb", "gain",
              "higher", "safe haven", "safe-haven", "rate cut", "dovish", "weaker dollar", "weak dollar",
              "bullish", "rebound", "recover", "haven demand", "upside"]
BEAR_WORDS = ["fall", "falls", "drop", "slump", "plunge", "decline", "hawkish", "stronger dollar",
              "strong dollar", "rate hike", "slide", "lower", "tumble", "sell-off", "selloff", "pressure",
              "bearish", "retreat", "downside", "loses", "slips", "weigh"]


def headline_score(title: str) -> int:
    t = title.lower()
    b = sum(w in t for w in BULL_WORDS)
    r = sum(w in t for w in BEAR_WORDS)
    return 1 if b > r else -1 if r > b else 0


@st.cache_data(ttl=300, show_spinner=False)
def google_news(query: str, limit: int = 20):
    url = ("https://news.google.com/rss/search?q=" + quote_plus(query + " when:2d")
           + "&hl=en-US&gl=US&ceid=US:en")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:  # noqa: BLE001
        return [], str(e)
    items = []
    for it in root.iter("item"):
        title = htmlmod.unescape(it.findtext("title", "") or "")
        src = it.findtext("source", "") or ""
        if src and title.endswith(f" - {src}"):
            title = title[: -len(src) - 3]
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:  # noqa: BLE001
            dt = None
        items.append({"title": title, "url": it.findtext("link", "#"), "source": src,
                      "dt": dt, "score": headline_score(title)})
    items.sort(key=lambda x: x["dt"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items[:limit], None


def render_google_items(items):
    for it in items:
        cls = "bull" if it["score"] > 0 else "bear" if it["score"] < 0 else "neut"
        lbl = "Bullish tone" if it["score"] > 0 else "Bearish tone" if it["score"] < 0 else "Neutral"
        when = it["dt"].strftime("%d %b, %H:%M UTC") if it["dt"] else ""
        st.markdown(
            f"""<div class="card"><span class="badge {cls}">{lbl}</span>
<span class="meta">{htmlmod.escape(it['source'])} · {when}</span>
<h4><a href="{it['url']}" target="_blank">{htmlmod.escape(it['title'])}</a></h4></div>""",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Daily history storage (SQLite) + comparison
# ----------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_history.db")
DB_COLS = ("gold", "wti", "fed_funds", "y10")


def db_conn():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS daily_rates (
            date TEXT PRIMARY KEY, gold REAL, wti REAL, fed_funds REAL, y10 REAL, updated_at TEXT)"""
    )
    return con


def save_daily(series: dict, days: int = 90) -> None:
    """Upsert the latest `days` daily values per series. Existing values are kept if a new one is missing."""
    frames = [df.tail(days).drop_duplicates("date").set_index("date")["value"].rename(col)
              for col, df in series.items() if df is not None]
    if not frames:
        return
    merged = pd.concat(frames, axis=1).sort_index()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = db_conn()
    for d, row in merged.iterrows():
        vals = [None if pd.isna(row.get(c)) else float(row[c]) for c in DB_COLS]
        con.execute(
            """INSERT INTO daily_rates(date,gold,wti,fed_funds,y10,updated_at) VALUES(?,?,?,?,?,?)
               ON CONFLICT(date) DO UPDATE SET
                 gold=COALESCE(excluded.gold,gold), wti=COALESCE(excluded.wti,wti),
                 fed_funds=COALESCE(excluded.fed_funds,fed_funds), y10=COALESCE(excluded.y10,y10),
                 updated_at=excluded.updated_at""",
            (d.strftime("%Y-%m-%d"), *vals, now),
        )
    con.commit()
    con.close()


def load_history() -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql("SELECT * FROM daily_rates ORDER BY date", con, parse_dates=["date"])
    con.close()
    return df


def summary_table(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("gold", "Gold"), ("wti", "WTI Oil"), ("fed_funds", "Fed Funds %"), ("y10", "US 10Y %")]:
        s = h[col].dropna()
        if len(s) < 2:
            continue
        last, prev = s.iloc[-1], s.iloc[-2]
        r = {"Metric": label, "Latest": last, "Prev day": prev, "1D chg": last - prev,
             "1D %": (last / prev - 1) * 100 if prev else None}
        for n in (5, 20):
            if len(s) > n:
                r[f"{n}D ago"] = s.iloc[-1 - n]
                r[f"{n}D chg"] = last - s.iloc[-1 - n]
        rows.append(r)
    return pd.DataFrame(rows)


def daily_compare(h: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    d = h.set_index("date")[list(DB_COLS)].ffill()
    out = pd.DataFrame(index=d.index)
    for c, lbl in [("gold", "Gold"), ("wti", "WTI"), ("fed_funds", "Fed %"), ("y10", "10Y %")]:
        out[lbl] = d[c]
        out[f"{lbl} chg"] = d[c].diff()
    out["Gold chg %"] = d["gold"].pct_change() * 100
    return out.tail(n).sort_index(ascending=False).reset_index()


# ----------------------------------------------------------------------------
# Rule-based gold bias (transparent, not a prediction)
# ----------------------------------------------------------------------------
def compute_bias(gold, y10, ffr, av_gold, gnews_items):
    f = []

    def add(name, score, detail):
        f.append({"Factor": name, "Score": score, "Detail": detail})

    if gold is not None and len(gold) >= 20:
        last, sma = gold["value"].iloc[-1], gold["value"].tail(20).mean()
        add("Gold trend vs 20D average", 1 if last > sma else -1, f"{last:,.1f} vs {sma:,.1f}")
    if y10 is not None and len(y10) >= 6:
        chg = y10["value"].iloc[-1] - y10["value"].iloc[-6]
        add("10Y yield, 5D change", -1 if chg > 0.05 else 1 if chg < -0.05 else 0,
            f"{chg:+.2f} pts (rising yields weigh on gold)")
    if ffr is not None and len(ffr) >= 21:
        chg = ffr["value"].iloc[-1] - ffr["value"].iloc[-21]
        add("Fed funds, 20D change", 1 if chg < -0.1 else -1 if chg > 0.1 else 0,
            f"{chg:+.2f} pts (cuts support gold)")
    if av_gold and "feed" in av_gold and av_gold["feed"]:
        sc = [a.get("overall_sentiment_score", 0) for a in av_gold["feed"][:30]]
        avg = sum(sc) / len(sc)
        add("Alpha Vantage gold news sentiment", 1 if avg > 0.15 else -1 if avg < -0.15 else 0, f"avg {avg:+.3f}")
    if gnews_items:
        avg = sum(i["score"] for i in gnews_items) / len(gnews_items)
        add("Google News headline tone", 1 if avg > 0.2 else -1 if avg < -0.2 else 0,
            f"{len(gnews_items)} headlines, avg {avg:+.2f}")
    total = sum(x["Score"] for x in f)
    n = len(f)
    label = "Bullish" if n and total >= max(2, n * 0.4) else "Bearish" if n and total <= -max(2, n * 0.4) else "Neutral / Mixed"
    return label, total, pd.DataFrame(f)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><h1>🥇 Gold Trading Analyst</h1>
<p>XAUUSD news, macro drivers and sentiment in one place · Updated {datetime.now():%d %b %Y %H:%M}</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh all data"):
        st.cache_data.clear()
        st.rerun()
    max_items = st.slider("Alpha Vantage articles per feed", 5, 50, 15)
    chart_days = st.slider("Chart history (days)", 30, 730, 180)
    st.caption(
        "Alpha Vantage free tier allows ~25 requests/day. Results are cached for 10 minutes."
    )
    if not API_KEY:
        st.error("Set ALPHAVANTAGE_API_KEY in .streamlit/secrets.toml")

tab_dash, tab_out, tab_hist, tab_av, tab_gn, tab_fx, tab_bond = st.tabs(
    ["📊 Market Dashboard", "🧭 Next Move Outlook", "🗓️ Daily Comparison", "📰 Sentiment News",
     "🔎 Google News", "🌐 FXStreet", "🏦 Bond Rates"]
)

# ---- Dashboard -------------------------------------------------------------
with tab_dash:
    gold = av_series_df(av_get({"function": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily"}))
    y10 = av_series_df(av_get({"function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year"}))
    wti = av_series_df(av_get({"function": "WTI", "interval": "daily"}))
    ffr = av_series_df(av_get({"function": "FEDERAL_FUNDS_RATE", "interval": "daily"}))

    m1, m2, m3, m4 = st.columns(4)
    delta_metric(m1, "Gold (XAU)", gold, " $")
    delta_metric(m2, "US 10Y Yield", y10, " %")
    delta_metric(m3, "WTI Crude", wti, " $")
    delta_metric(m4, "Fed Funds Rate", ffr, " %")

    if all(d is None for d in (gold, y10, wti, ffr)):
        st.warning("No market series loaded. Check API key / rate limit.")

    c1, c2 = st.columns(2)
    with c1:
        if gold is not None:
            line_chart(gold, "Gold – daily", "#ffd24d", chart_days)
        if wti is not None:
            line_chart(wti, "WTI Crude – daily", "#4da6ff", chart_days)
    with c2:
        if y10 is not None:
            line_chart(y10, "US 10Y Treasury Yield", "#ff6b6b", chart_days)
        if ffr is not None:
            line_chart(ffr, "Federal Funds Rate", "#9b8cff", chart_days)

# ---- Alpha Vantage news ----------------------------------------------------
with tab_av:
    sub = st.tabs(list(AV_NEWS_FEEDS.keys()))
    for t, (name, p) in zip(sub, AV_NEWS_FEEDS.items()):
        with t:
            render_av_news(av_get({"function": "NEWS_SENTIMENT", **p}), max_items)

# ---- FXStreet --------------------------------------------------------------
with tab_fx:
    a, b = st.columns(2)
    with a:
        st.subheader("Top News (Main Page)")
        render_links(*fx_main())
    with b:
        st.subheader("Commodities – Top 3")
        render_links(*fx_commodities())
        st.subheader("Latest News – Top 5")
        render_links(*fx_generic())

# ---- Bonds -----------------------------------------------------------------
with tab_bond:
    st.subheader("Bond Rates")
    df_b, err_b = bloomberg_rates()
    if df_b is not None:
        st.dataframe(df_b, use_container_width=True)
    else:
        st.warning(f"Bloomberg blocked/unavailable: {err_b}")
        st.caption("Showing US Treasury yield from Alpha Vantage instead.")
        if y10 is not None:
            line_chart(y10, "US 10Y Treasury Yield", "#ff6b6b", chart_days, key="bond_tab_10y")
    st.link_button("Open Bloomberg Rates & Bonds", "https://www.bloomberg.com/markets/rates-bonds")

# ---- Google News -----------------------------------------------------------
with tab_gn:
    st.caption("Google News RSS, last 48h, newest first. Edit GOOGLE_QUERIES in the code to change topics.")
    custom_q = st.text_input("Custom Google News query (optional)", "")
    if custom_q.strip():
        items, err = google_news(custom_q.strip())
        st.warning(err) if err else render_google_items(items)
        st.divider()
    subs = st.tabs(list(GOOGLE_QUERIES.keys()))
    for t, (name, q) in zip(subs, GOOGLE_QUERIES.items()):
        with t:
            items, err = google_news(q)
            if err:
                st.warning(f"Could not load: {err}")
            elif not items:
                st.info("No articles returned.")
            else:
                render_google_items(items)

# ---- Daily comparison (stored in SQLite) -----------------------------------
with tab_hist:
    try:
        save_daily({"gold": gold, "wti": wti, "fed_funds": ffr, "y10": y10})
        hist = load_history()
    except Exception as e:  # noqa: BLE001
        hist = pd.DataFrame()
        st.error(f"Database error: {e}")
    if hist.empty:
        st.info("No stored data yet. Load market data first (check API key / rate limit).")
    else:
        st.caption(f"Stored in `{DB_PATH}` · {len(hist)} days · updated every time the app loads "
                   "(one row per date, so it builds up history automatically).")
        st.subheader("Latest vs previous")
        summ = summary_table(hist)
        num = {c: st.column_config.NumberColumn(format="%.2f") for c in summ.columns if c != "Metric"}
        for c in [c for c in summ.columns if "chg" in c]:
            num[c] = st.column_config.NumberColumn(format="%+.2f")
        num["1D %"] = st.column_config.NumberColumn(format="%+.2f%%")
        st.dataframe(summ, use_container_width=True, hide_index=True, column_config=num)

        st.subheader("Day-by-day comparison")
        days_n = st.slider("Days to show", 5, 90, 20, key="cmp_days")
        cmp_df = daily_compare(hist, days_n)
        cfg = {"date": st.column_config.DateColumn("Date", format="DD MMM YYYY")}
        for c in cmp_df.columns:
            if c != "date":
                cfg[c] = st.column_config.NumberColumn(format="%+.2f" if "chg" in c else "%.2f")
        st.dataframe(cmp_df, use_container_width=True, hide_index=True, column_config=cfg)
        st.download_button("Download stored history (CSV)", hist.to_csv(index=False),
                           "gold_history.csv", "text/csv")

# ---- Next move outlook -----------------------------------------------------
with tab_out:
    gnews_all, seen = [], set()
    for q in GOOGLE_QUERIES.values():
        its, _ = google_news(q)
        for it in its:
            if it["title"] not in seen:
                seen.add(it["title"])
                gnews_all.append(it)
    av_gold = av_get({"function": "NEWS_SENTIMENT", **AV_NEWS_FEEDS["Gold"]})
    label, total, fdf = compute_bias(gold, y10, ffr, av_gold, gnews_all)
    o1, o2 = st.columns(2)
    o1.metric("Short-term bias", label)
    o2.metric("Net score", f"{total:+d}", help="Sum of factor scores (each -1, 0 or +1)")
    if fdf.empty:
        st.info("Not enough data to compute a bias yet.")
    else:
        st.dataframe(fdf, use_container_width=True, hide_index=True)
    st.subheader("Strongest headlines")
    tone = [i for i in gnews_all if i["score"] != 0][:10]
    render_google_items(tone) if tone else st.info("No strongly directional headlines right now.")
    st.caption("This is a simple rule-based read of trend, yields, Fed rate and news tone. "
               "Headline tone comes from keyword matching, so it is rough. Use it as one input, not a signal.")

st.caption("For information only. Not financial advice.")