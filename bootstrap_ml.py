"""
bootstrap_ml.py — Dual-Layer ML: Camada A (preço, 20 anos) + Camada B (fundamentais, 7 anos).

O UNIVERSO está hardcoded directamente neste ficheiro (S&P 500, Nasdaq 100, STOXX 200,
FTSE 100, carteira e watchlist do utilizador). Zero dependências externas, zero scraping.
~830 tickers únicos após deduplicação.

MODO AUTOMÁTICO (agendado pelo bot, corre às 02:00 UTC todos os dias):
    - Janela = [hoje - anos_config, ontem]
    - Registos fora da janela são eliminados automaticamente
    - Retreina ambas as camadas

MODO MANUAL (Railway CLI):
    python bootstrap_ml.py                          # tudo com defaults
    python bootstrap_ml.py --algo xgb
    python bootstrap_ml.py --layer price
    python bootstrap_ml.py --layer fund
    python bootstrap_ml.py --skip-backfill          # só treino
    python bootstrap_ml.py --force-full             # refaz backfill completo

MODO COLAB — backfill em batches (Google Drive como persistência):
    # Sessão 1 — tickers 0..200
    python bootstrap_ml.py --layer price --slice 0 200 --drive-dir /content/drive/MyDrive/DipRadar
    # Sessão 2 — tickers 200..400
    python bootstrap_ml.py --layer price --slice 200 400 --drive-dir /content/drive/MyDrive/DipRadar
    # Sessão 3 — tickers 400..600
    python bootstrap_ml.py --layer price --slice 400 600 --drive-dir /content/drive/MyDrive/DipRadar
    # Sessão 4 — tickers 600..800
    python bootstrap_ml.py --layer price --slice 600 800 --drive-dir /content/drive/MyDrive/DipRadar
    # Sessão 5 — resto
    python bootstrap_ml.py --layer price --slice 800 999 --drive-dir /content/drive/MyDrive/DipRadar
    # Sessão final — treino com o Parquet completo
    python bootstrap_ml.py --layer price --skip-backfill --drive-dir /content/drive/MyDrive/DipRadar

    O argumento --slice START END limita o backfill aos tickers UNIVERSE[START:END].
    O Parquet no Drive é acumulado incrementalmente — cada sessão retoma de onde parou.
    --drive-dir define o directório onde os Parquets e .pkl são guardados/lidos.

OUTPUT:
    <data_dir>/dip_model_price.pkl    ← Camada A (só técnicas)
    <data_dir>/dip_model_stage1.pkl   ← Camada B stage 1 (win vs no-win)
    <data_dir>/dip_model_stage2.pkl   ← Camada B stage 2 (win40 vs win20)
    <data_dir>/ml_training_price.parquet
    <data_dir>/ml_training_fund.parquet
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bootstrap_ml] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bootstrap_ml")

YEARS_PRICE = 20
YEARS_FUND  = 7

# ── Universo hardcoded ─────────────────────────────────────────────────────────

_SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB",
    "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP",
    "AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL",
    "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","AZO","AVB","AVY",
    "AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB",
    "BLK","BX","BA","BCR","BWA","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS",
    "CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW",
    "CE","COR","CNC","CNP","CF","CHRW","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI",
    "CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG",
    "COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRWD",
    "CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DE","DAL","XRAY","DVN","DXCM","FANG",
    "DLR","DFS","DG","DLTR","D","DPZ","DOV","DHI","DTE","DUK","DD","EMN","ETN","EBAY",
    "ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR",
    "ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO",
    "FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX",
    "BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD",
    "GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT",
    "HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX",
    "ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV",
    "IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS",
    "KMB","KIM","KMI","KKR","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN",
    "LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR",
    "MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM",
    "MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS",
    "MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN",
    "NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC",
    "ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PANW","PARA","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR",
    "PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O",
    "REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV",
    "SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO",
    "TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TPL","TXT","TMO","TJX",
    "TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP",
    "UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS",
    "VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC",
    "WFC","WELL","WST","WDC","WRK","WY","WHR","WMB","WTW","WYNN","XEL","XYL","YUM",
    "ZBRA","ZBH","ZTS",
]

_NASDAQ100 = [
    "ADBE","ADP","ABNB","ALGN","GOOGL","GOOG","AMZN","AMD","AEP","AMGN","ADI","ANSS",
    "AAPL","AMAT","ASML","AZN","TEAM","ADSK","BIDU","BIIB","BKNG","AVGO","CDNS","CDW",
    "CHTR","CTAS","CSCO","CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX",
    "DDOG","DXCM","FANG","DLTR","EBAY","EA","EXC","FAST","FTNT","GILD","GFS","HON",
    "IDXX","ILMN","INTC","INTU","ISRG","JD","KDP","KLAC","KHC","LRCX","LIN","LULU",
    "MAR","MRVL","MELI","META","MCHP","MU","MSFT","MRNA","MDLZ","MDB","MNST","NFLX",
    "NVDA","NXPI","ORLY","ODFL","ON","PCAR","PANW","PAYX","PYPL","PDD","PEP","QCOM",
    "REGN","ROP","ROST","SBUX","SGEN","SIRI","SNPS","TSLA","TXN","TMUS","TTWO","VRSK",
    "VRTX","WBA","WBD","XEL","ZM","ZS",
]

_STOXX200 = [
    # Alemanha (DAX)
    "SAP.DE","SIE.DE","ALV.DE","MRK.DE","BMW.DE","MBG.DE","BAS.DE","BAYN.DE","DTE.DE",
    "DBK.DE","VOW3.DE","ADS.DE","HNR1.DE","RWE.DE","HEI.DE","ENR.DE","EOAN.DE","CON.DE",
    "FRE.DE","ZAL.DE","LIN.DE","MUV2.DE","DHER.DE","SHL.DE","MTX.DE","1COV.DE","BOSS.DE",
    "HAB.DE","HFG.DE","VNA.DE","BEI.DE","SYMB.DE","SY1.DE","KGX.DE","AIXA.DE","CARL.DE",
    "QIA.DE","EVD.DE","DHL.DE","PUM.DE",
    # França (CAC 40)
    "MC.PA","OR.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","ACA.PA","SGO.PA","SU.PA",
    "DG.PA","RI.PA","CS.PA","KER.PA","ATO.PA","CAP.PA","VIE.PA","LR.PA","DSY.PA",
    "HO.PA","ML.PA","ORA.PA","ENGI.PA","STM.PA","SAF.PA","BN.PA","SG.PA","EL.PA",
    "RMS.PA","WLN.PA","CA.PA","ATO.PA","TEP.PA","RNO.PA","URW.PA","PUB.PA",
    # Países Baixos
    "ASML.AS","HEIA.AS","PHIA.AS","REN.AS","UNA.AS","INGA.AS","ABN.AS","NN.AS",
    "RAND.AS","AKZA.AS","IMCD.AS","WKL.AS","ADY.AS","BESI.AS","LIGHT.AS",
    # Suíça
    "NESN.SW","NOVN.SW","ROG.SW","UBSG.SW","CSGN.SW","ABBN.SW","ZURN.SW","GIVN.SW",
    "LONN.SW","CFR.SW","SREN.SW","SCMN.SW","BAER.SW","ALC.SW","GEBN.SW","SIKA.SW",
    # Espanha
    "ITX.MC","SAN.MC","TEF.MC","BBVA.MC","IBE.MC","AMS.MC","REP.MC","CABK.MC",
    "BKT.MC","GRF.MC","MAP.MC","ENG.MC","NTGY.MC","MTS.MC",
    # Itália
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","STM.MI","LDO.MI","MB.MI","FCA.MI",
    "PRY.MI","RACE.MI","MONC.MI","BMED.MI","PST.MI",
    # Suécia
    "VOLV-B.ST","ERIC-B.ST","ATCO-A.ST","SEB-A.ST","SHB-A.ST","SWED-A.ST","INVE-B.ST",
    "SSAB-A.ST","SAND.ST","SKF-B.ST","HM-B.ST","ALFA.ST","NIBE-B.ST","TELIA.ST",
    # Dinamarca
    "NOVO-B.CO","ORSTED.CO","CARL-B.CO","DSV.CO","COLO-B.CO","DEMANT.CO","GN.CO",
    # Noruega
    "EQNR.OL","DNB.OL","MOWI.OL","TEL.OL","YAR.OL","NHY.OL","ORKLA.OL",
    # Finlândia
    "NOKIA.HE","FORTUM.HE","NESTE.HE","KNEBV.HE","STERV.HE",
    # Bélgica
    "UCB.BR","ABI.BR","SOLB.BR","AGS.BR","GLPG.BR",
    # Irlanda
    "CRH.I","KYGA.I","AIB.I","BIRG.I",
    # Portugal
    "EDP.LS","GALP.LS","BCP.LS","EDPR.LS",
    # Áustria
    "VIG.VI","OMV.VI","EBS.VI",
    # Polónia
    "PKN.WA","PKO.WA","PZU.WA","DNP.WA","LPP.WA",
    # Luxemburgo / outros
    "APAM.AS","TKWY.AS",
]

_FTSE100 = [
    "AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","DGE.L","REL.L",
    "BATS.L","LLOY.L","BARC.L","VOD.L","BT-A.L","NG.L","SSE.L","IMB.L","CRH.L",
    "LSEG.L","NWG.L","PRU.L","EXPN.L","ABF.L","FERG.L","TSCO.L","CPG.L","RKT.L",
    "WPP.L","IAG.L","EZJ.L","AAL.L","ANTO.L","BHP.L","GLEN.L","EVR.L","MNDI.L",
    "SMDS.L","SMT.L","III.L","LAND.L","SGRO.L","HMSO.L","BLND.L","DLN.L","PSN.L",
    "BWY.L","TW.L","BA.L","RR.L","AUTO.L","OCDO.L","MKS.L","NEXT.L","JD.L","SPX.L",
    "RS1.L","DCC.L","SKG.L","IMI.L","PHNX.L","LGEN.L","AV.L","ADM.L","STAN.L",
    "INVP.L","RTO.L","SBRY.L","MRO.L","SVT.L","UU.L","PNN.L","HLN.L","BDEV.L",
    "BME.L","CRDA.L","ENT.L","FLTR.L","KGF.L","LGEN.L","MNG.L","SGRO.L","SN.L",
    "SPX.L","WEIR.L","WTB.L","XP.L",
]

_CARTEIRA = [
    "NVO","ADBE","UBER","MSFT","PINS","ADP","CRM","VICI","CRWD","PLTR","NOW","DUOL","EUNL.DE",
]

_WATCHLIST = [
    "O","MDT","ABBV","LMT","RTX","PANW","TSM","AVGO","ALV.DE","IEMA",
]


def _load_universe() -> list[str]:
    """
    Universo ML completo hardcoded — sem scraping, sem dependências externas.
    S&P 500 (~504) + Nasdaq 100 (~103) + STOXX 200 (~200) + FTSE 100 (~85)
    + carteira + watchlist, deduplicado e sem ETFs.
    """
    ETF_BLACKLIST = {
        "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","LQD","EEM","VWO","EFA",
        "VEA","VTI","VNQ","XLF","XLK","XLE","XLV","XLI","XLU","XLB","XLP","XLY",
        "XLRE","XLC","IEMA","EUNL.DE","SCHD","JEPI","JEPQ","SOXL","TQQQ","SQQQ",
    }
    all_tickers: list[str] = []
    seen: set[str] = set()
    for lst in (_SP500, _NASDAQ100, _STOXX200, _FTSE100, _CARTEIRA, _WATCHLIST):
        for t in lst:
            t = t.strip()
            if t and t not in seen and t not in ETF_BLACKLIST:
                all_tickers.append(t)
                seen.add(t)

    log.info(f"[universe] Universo total: {len(all_tickers)} tickers (hardcoded, zero scraping)")
    return all_tickers


def _window(years: int) -> tuple[date, date]:
    end   = date.today() - timedelta(days=1)
    start = end.replace(year=end.year - years)
    return start, end


def _normalize_history_index(hist: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone do índice do yfinance para evitar erros de comparação."""
    if hist.empty:
        return hist
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        hist.index = idx.tz_localize(None)
    return hist.sort_index()


# ── Features da Camada A (melhoradas) ─────────────────────────────────────────
#
# Novas vs versão anterior:
#   + dist_ma50        — distância % à MA50 (dip técnico real vs ruído)
#   + dist_ma200       — distância % à MA200 (bear market vs correcção)
#   + ret_1m           — momentum do mês anterior ao dip
#   + ret_3m_prior     — momentum 3 meses antes do dip
#   + month            — sazonalidade (Set/Out historicamente piores)
#   + high_52w_pct     — % abaixo do máximo de 52 semanas
#
# Label: alpha vs SPY — um dip que sobe 20% com SPY +25% é NEUTRAL, não WIN
#   alpha_6m = ret_6m - ret_spy_6m   (ou 3m se 6m não disponível)
#   WIN_40 → alpha >= 30%
#   WIN_20 → alpha >= 15%
#   NEUTRAL → alpha >= -10%
#   LOSS    → alpha < -10%

FEATURES_PRICE: list[str] = [
    # originais
    "rsi", "drawdown_pct", "change_day_pct",
    "beta", "spy_change", "sector_etf_change",
    "volume_ratio", "atr_pct",
    # novas
    "dist_ma50", "dist_ma200",
    "ret_1m", "ret_3m_prior",
    "month", "high_52w_pct",
]

FEATURES_FUND: list[str] = [
    "rsi", "drawdown_pct", "change_day_pct",
    "pe_ratio", "pb_ratio", "fcf_yield", "analyst_upside",
    "revenue_growth", "gross_margin",
    "debt_to_equity", "beta", "short_pct",
    "spy_change", "sector_etf_change", "earnings_days",
    "market_cap_b", "dip_score",
]


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def calc_atr_pct(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low   = hist["High"] - hist["Low"]
    high_close = (hist["High"] - hist["Close"].shift()).abs()
    low_close  = (hist["Low"]  - hist["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return (atr / hist["Close"]) * 100


def calc_volume_ratio(hist: pd.DataFrame, period: int = 20) -> pd.Series:
    avg = hist["Volume"].rolling(period).mean()
    return hist["Volume"] / (avg + 1e-9)


def outcome_label_alpha(stock_ret: float, spy_ret: float) -> str:
    """Label ajustado ao SPY — mede alpha real, não retorno absoluto."""
    alpha = stock_ret - spy_ret
    if   alpha >= 30:  return "WIN_40"
    elif alpha >= 15:  return "WIN_20"
    elif alpha >= -10: return "NEUTRAL"
    else:              return "LOSS_15"


def outcome_label(ret: float) -> str:
    """Label absoluto — usado na Camada B (fundamentais)."""
    if   ret >= 40:  return "WIN_40"
    elif ret >= 20:  return "WIN_20"
    elif ret >= -15: return "NEUTRAL"
    else:            return "LOSS_15"


def get_price_near(hist: pd.DataFrame, target: date) -> float | None:
    for d in range(-3, 6):
        check = target + timedelta(days=d)
        m = hist[hist.index.date == check]
        if not m.empty:
            return float(m["Close"].iloc[0])
    return None


def safe_float(val, default=None):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def simple_dip_score(r: dict) -> float:
    score = 50.0
    rsi = r.get("rsi") or 50
    if   rsi < 25: score += 20
    elif rsi < 35: score += 12
    elif rsi < 45: score += 5
    ddp = r.get("drawdown_pct") or 0
    if   ddp <= -40: score += 20
    elif ddp <= -25: score += 12
    elif ddp <= -15: score += 7
    elif ddp <= -10: score += 3
    chg = r.get("change_day_pct") or 0
    if   chg <= -8:  score += 15
    elif chg <= -5:  score += 9
    elif chg <= -3:  score += 4
    pe = r.get("pe_ratio") or 20
    if pe > 0:
        if   pe < 12: score += 10
        elif pe < 18: score += 5
        elif pe > 50: score -= 5
    up = r.get("analyst_upside") or 0
    if   up > 40: score += 10
    elif up > 20: score += 5
    return min(max(score, 0), 100)


# ── Backfill — Camada A (preço puro, 20 anos) ─────────────────────────────────

def backfill_price(
    start: date,
    end: date,
    tickers: list[str],
    dip_thresh: float = 0.04,
    max_per_ticker: int = 10,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance não instalado")
        sys.exit(1)

    existing_keys: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        existing_keys = set(zip(existing_df["symbol"].astype(str), existing_df["alert_date"].astype(str)))

    start_str = start.isoformat()
    fetch_end = min(end + timedelta(days=200), date.today()).isoformat()
    start_ts  = pd.Timestamp(start)
    end_ts    = pd.Timestamp(end)

    log.info(f"[CamadaA] {start_str} → {end.isoformat()} | {len(tickers)} tickers | existentes: {len(existing_keys)}")

    spy_hist = yf.Ticker("SPY").history(start=start_str, end=fetch_end, interval="1d")
    spy_hist = _normalize_history_index(spy_hist)
    spy_hist["spy_ret"] = spy_hist["Close"].pct_change() * 100
    spy_map  = {d.date(): float(r) for d, r in spy_hist["spy_ret"].items()}
    spy_close = {d.date(): float(c) for d, c in spy_hist["Close"].items()}

    all_alerts: list[dict] = []

    for i, ticker in enumerate(tickers):
        try:
            hist = yf.Ticker(ticker).history(start=start_str, end=fetch_end, interval="1d")
            hist = _normalize_history_index(hist)
            if hist.empty or len(hist) < 60:
                continue

            # ── indicadores técnicos ───────────────────────────────────────
            hist["rsi"]       = calc_rsi(hist["Close"])
            hist["ret_1d"]    = hist["Close"].pct_change() * 100
            hist["atr_pct"]   = calc_atr_pct(hist)
            hist["vol_ratio"] = calc_volume_ratio(hist)

            # drawdown desde máximo rolante 252d
            roll_max     = hist["Close"].rolling(252, min_periods=30).max()
            hist["ddp"]  = (hist["Close"] - roll_max) / roll_max * 100

            # distância às médias móveis
            hist["ma50"]      = hist["Close"].rolling(50, min_periods=20).mean()
            hist["ma200"]     = hist["Close"].rolling(200, min_periods=60).mean()
            hist["dist_ma50"] = (hist["Close"] - hist["ma50"])  / hist["ma50"]  * 100
            hist["dist_ma200"]= (hist["Close"] - hist["ma200"]) / hist["ma200"] * 100

            # % abaixo do máximo de 52 semanas (diferente do drawdown: usa só 252d)
            high_52w           = hist["Close"].rolling(252, min_periods=30).max()
            hist["high_52w_pct"] = (hist["Close"] - high_52w) / high_52w * 100

            # momentum prévio ao dip (retorno 1m e 3m antes)
            hist["ret_1m"]     = hist["Close"].pct_change(21)  * 100   # ~1 mês
            hist["ret_3m"]     = hist["Close"].pct_change(63)  * 100   # ~3 meses

            # beta rolante 252d vs SPY
            spy_aligned = pd.Series(spy_map).reindex([d.date() for d in hist.index], fill_value=np.nan)
            spy_aligned.index = hist.index
            cov = hist["ret_1d"].rolling(252).cov(spy_aligned)
            var = spy_aligned.rolling(252).var()
            hist["beta_roll"] = (cov / (var + 1e-9)).clip(-3, 5)

            mask = (
                (hist["ret_1d"] <= -(dip_thresh * 100)) &
                (hist["rsi"] < 55) &
                (hist.index >= start_ts) &
                (hist.index <= end_ts)
            )
            dip_days = hist.loc[mask]
            if dip_days.empty:
                continue

            selected = []
            last_dt = None
            for dt, row in dip_days.iterrows():
                alert_date = dt.date()
                if (ticker, alert_date.isoformat()) in existing_keys:
                    continue
                if last_dt is None or (alert_date - last_dt).days >= 20:
                    selected.append((dt, row))
                    last_dt = alert_date
                if len(selected) >= max_per_ticker:
                    break

            for dt, row in selected:
                alert_date = dt.date()
                spy_chg    = spy_map.get(alert_date, 0.0)
                hist_after = hist[hist.index.date > alert_date]
                if hist_after.empty:
                    continue

                entry = float(row["Close"])

                # retorno do ticker
                p3m = get_price_near(hist_after, alert_date + timedelta(days=91))
                p6m = get_price_near(hist_after, alert_date + timedelta(days=182))
                if p3m is None and p6m is None:
                    continue
                r3m = (p3m - entry) / entry * 100 if p3m else None
                r6m = (p6m - entry) / entry * 100 if p6m else None

                # retorno do SPY no mesmo período (para calcular alpha)
                spy_dates = sorted(spy_close.keys())
                def spy_ret_period(days: int) -> float | None:
                    target = alert_date + timedelta(days=days)
                    # preço SPY no dia do alerta
                    spy_entry = spy_close.get(alert_date)
                    if spy_entry is None:
                        # procura vizinho
                        for d in range(-3, 6):
                            spy_entry = spy_close.get(alert_date + timedelta(days=d))
                            if spy_entry:
                                break
                    if spy_entry is None:
                        return None
                    # preço SPY ~days depois
                    for d in range(-3, 6):
                        sp = spy_close.get(target + timedelta(days=d))
                        if sp is not None:
                            return (sp - spy_entry) / spy_entry * 100
                    return None

                spy_r3m = spy_ret_period(91)
                spy_r6m = spy_ret_period(182)

                # escolhe janela principal (6m preferido)
                ref        = r6m if r6m is not None else r3m
                spy_ref    = spy_r6m if r6m is not None else spy_r3m
                if ref is None:
                    continue
                spy_ref = spy_ref if spy_ref is not None else 0.0

                all_alerts.append({
                    "symbol":            ticker,
                    "alert_date":        alert_date.isoformat(),
                    "price":             round(entry, 2),
                    # features originais
                    "rsi":               round(safe_float(row["rsi"], 50), 1),
                    "drawdown_pct":      round(safe_float(row["ddp"], 0), 2),
                    "change_day_pct":    round(float(row["ret_1d"]), 2),
                    "beta":              round(safe_float(row["beta_roll"], 1.0), 2),
                    "atr_pct":           round(safe_float(row["atr_pct"], 1.0), 2),
                    "volume_ratio":      round(safe_float(row["vol_ratio"], 1.0), 2),
                    "spy_change":        round(spy_chg, 2),
                    "sector_etf_change": round(spy_chg * 0.9, 2),
                    # novas features
                    "dist_ma50":         round(safe_float(row["dist_ma50"], 0.0), 2),
                    "dist_ma200":        round(safe_float(row["dist_ma200"], 0.0), 2),
                    "ret_1m":            round(safe_float(row["ret_1m"], 0.0), 2),
                    "ret_3m_prior":      round(safe_float(row["ret_3m"], 0.0), 2),
                    "month":             alert_date.month,
                    "high_52w_pct":      round(safe_float(row["high_52w_pct"], 0.0), 2),
                    # outcomes
                    "return_3m":         round(r3m, 2) if r3m is not None else None,
                    "return_6m":         round(r6m, 2) if r6m is not None else None,
                    "spy_return_ref":    round(spy_ref, 2),
                    # label alpha vs SPY
                    "outcome_label":     outcome_label_alpha(ref, spy_ref),
                })

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(tickers)}] {len(all_alerts)} alertas")
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  ERRO {ticker}: {e}")

    log.info(f"[CamadaA] {len(all_alerts)} novos alertas")
    return pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame()


# ── Backfill — Camada B (fundamentais, 7 anos) ───────────────────────────────

def backfill_fund(
    start: date,
    end: date,
    tickers: list[str],
    dip_thresh: float = 0.04,
    max_per_ticker: int = 8,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance não instalado")
        sys.exit(1)

    existing_keys: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        existing_keys = set(zip(existing_df["symbol"].astype(str), existing_df["alert_date"].astype(str)))

    start_str = start.isoformat()
    fetch_end = min(end + timedelta(days=200), date.today()).isoformat()
    start_ts  = pd.Timestamp(start)
    end_ts    = pd.Timestamp(end)

    log.info(f"[CamadaB] {start_str} → {end.isoformat()} | {len(tickers)} tickers | existentes: {len(existing_keys)}")

    spy_hist = yf.Ticker("SPY").history(start=start_str, end=fetch_end, interval="1d")
    spy_hist = _normalize_history_index(spy_hist)
    spy_hist["spy_ret"] = spy_hist["Close"].pct_change() * 100
    spy_map = {d.date(): float(r) for d, r in spy_hist["spy_ret"].items()}

    all_alerts: list[dict] = []

    for i, ticker in enumerate(tickers):
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.history(start=start_str, end=fetch_end, interval="1d")
            hist = _normalize_history_index(hist)
            if hist.empty or len(hist) < 60:
                continue

            info = tk.info or {}
            hist["rsi"]    = calc_rsi(hist["Close"])
            hist["ret_1d"] = hist["Close"].pct_change() * 100
            roll_max       = hist["Close"].rolling(252, min_periods=30).max()
            hist["ddp"]    = (hist["Close"] - roll_max) / roll_max * 100

            pe     = safe_float(info.get("trailingPE") or info.get("forwardPE"))
            pb     = safe_float(info.get("priceToBook"))
            mcap   = safe_float(info.get("marketCap"), 0) / 1e9
            fcf    = safe_float(info.get("freeCashflow"))
            mc_raw = safe_float(info.get("marketCap"))
            fcfy   = (fcf / mc_raw * 100) if fcf and mc_raw else None
            revg   = safe_float(info.get("revenueGrowth"), 0) * 100
            gm     = safe_float(info.get("grossMargins"), 0) * 100
            de     = safe_float(info.get("debtToEquity"), 0) / 100
            beta   = safe_float(info.get("beta"), 1.0)
            short  = safe_float(info.get("shortPercentOfFloat"), 0) * 100
            tgt    = safe_float(info.get("targetMeanPrice"))
            cur    = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"), 1)
            upside = ((tgt - cur) / cur * 100) if tgt and cur else 0.0

            mask = (
                (hist["ret_1d"] <= -(dip_thresh * 100)) &
                (hist["rsi"] < 55) &
                (hist.index >= start_ts) &
                (hist.index <= end_ts)
            )
            dip_days = hist.loc[mask]
            if dip_days.empty:
                continue

            selected = []
            last_dt = None
            for dt, row in dip_days.iterrows():
                alert_date = dt.date()
                if (ticker, alert_date.isoformat()) in existing_keys:
                    continue
                if last_dt is None or (alert_date - last_dt).days >= 20:
                    selected.append((dt, row))
                    last_dt = alert_date
                if len(selected) >= max_per_ticker:
                    break

            for dt, row in selected:
                alert_date = dt.date()
                spy_chg    = spy_map.get(alert_date, 0.0)
                hist_after = hist[hist.index.date > alert_date]
                if hist_after.empty:
                    continue

                entry = float(row["Close"])
                p3m = get_price_near(hist_after, alert_date + timedelta(days=91))
                p6m = get_price_near(hist_after, alert_date + timedelta(days=182))
                if p3m is None and p6m is None:
                    continue

                r3m = (p3m - entry) / entry * 100 if p3m else None
                r6m = (p6m - entry) / entry * 100 if p6m else None
                ref = r6m if r6m is not None else r3m
                if ref is None:
                    continue

                feat: dict = {
                    "rsi":               round(safe_float(row["rsi"], 50), 1),
                    "drawdown_pct":      round(safe_float(row["ddp"], 0), 2),
                    "change_day_pct":    round(float(row["ret_1d"]), 2),
                    "pe_ratio":          round(pe, 1) if pe else None,
                    "pb_ratio":          round(pb, 2) if pb else None,
                    "fcf_yield":         round(fcfy, 4) if fcfy else None,
                    "analyst_upside":    round(upside, 1),
                    "revenue_growth":    round(revg, 2),
                    "gross_margin":      round(gm, 2),
                    "debt_to_equity":    round(de, 2),
                    "beta":              round(beta, 2),
                    "short_pct":         round(short, 2),
                    "spy_change":        round(spy_chg, 2),
                    "sector_etf_change": round(spy_chg * 0.9, 2),
                    "earnings_days":     90,
                    "market_cap_b":      round(mcap, 2),
                }
                feat["dip_score"]     = round(simple_dip_score(feat), 1)
                feat["symbol"]        = ticker
                feat["alert_date"]    = alert_date.isoformat()
                feat["price"]         = round(entry, 2)
                feat["return_3m"]     = round(r3m, 2) if r3m is not None else None
                feat["return_6m"]     = round(r6m, 2) if r6m is not None else None
                feat["outcome_label"] = outcome_label(ref)
                all_alerts.append(feat)

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(tickers)}] {len(all_alerts)} alertas")
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  ERRO {ticker}: {e}")

    log.info(f"[CamadaB] {len(all_alerts)} novos alertas")
    return pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame()


# ── Janela deslizante ─────────────────────────────────────────────────────────

def load_and_slide(
    parquet: Path,
    start: date,
    new_df: pd.DataFrame,
    skip_exit_on_empty: bool = False,
) -> pd.DataFrame:
    start_str = start.isoformat()
    if parquet.exists():
        existing = pd.read_parquet(parquet)
        rows_before = len(existing)
        existing["alert_date"] = existing["alert_date"].astype(str)
        existing = existing[existing["alert_date"] >= start_str].copy()
        purged = rows_before - len(existing)
        if purged > 0:
            log.info(f"🗑  Purgados {purged} registos fora da janela (< {start_str})")
    else:
        existing = pd.DataFrame()

    if new_df.empty and existing.empty:
        if skip_exit_on_empty:
            log.warning("Sem dados novos e Parquet inexistente — a saltar treino.")
            return pd.DataFrame()
        log.error("Sem dados — Parquet vazio e backfill sem resultados.")
        sys.exit(1)

    if new_df.empty:
        combined = existing
    elif existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    combined["alert_date"] = combined["alert_date"].astype(str)
    combined.drop_duplicates(subset=["symbol", "alert_date"], keep="last", inplace=True)
    combined.sort_values("alert_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.to_parquet(parquet, index=False)
    log.info(f"📦 Parquet: {len(combined)} registos → {parquet}")
    return combined


# ── Pipeline de treino ────────────────────────────────────────────────────────

def _build_pipeline(algo: str = "rf"):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif algo == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            log.warning("xgboost não instalado — a usar GradientBoosting")
            clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algo}")
    steps.append(("clf", clf))
    return Pipeline(steps)


def _train_layer(
    df: pd.DataFrame,
    features: list[str],
    pkl_s1: Path,
    pkl_s2: Path | None,
    algo: str,
    label: str,
) -> None:
    from sklearn.metrics import average_precision_score, classification_report

    if df.empty:
        log.warning(f"[{label}] DataFrame vazio — treino saltado.")
        return

    df2 = df[df["outcome_label"].notna()].copy()
    df2["target_s1"] = df2["outcome_label"].apply(lambda x: 1 if x in ("WIN_40", "WIN_20") else 0)
    if len(df2) < 30 or df2["target_s1"].sum() < 10:
        log.error(f"[{label}] Dados insuficientes: {len(df2)} linhas, {int(df2['target_s1'].sum())} wins")
        return

    for col in features:
        if col not in df2.columns:
            df2[col] = np.nan

    df2 = df2.sort_values("alert_date").reset_index(drop=True)
    split    = int(len(df2) * 0.80)
    train_df = df2.iloc[:split]
    test_df  = df2.iloc[split:]

    X_tr = train_df[features].values.astype(np.float32)
    y_tr = train_df["target_s1"].values
    X_te = test_df[features].values.astype(np.float32)
    y_te = test_df["target_s1"].values

    log.info(f"[{label}] {algo.upper()} | train={len(X_tr)} test={len(X_te)} wins={y_tr.sum()}")

    pipe = _build_pipeline(algo)
    if algo == "xgb":
        ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
        pipe.named_steps["clf"].set_params(scale_pos_weight=ratio)
    pipe.fit(X_tr, y_tr)

    probs  = pipe.predict_proba(X_te)[:, 1]
    y_pred = (probs >= 0.50).astype(int)
    auc_pr = average_precision_score(y_te, probs)

    log.info(f"[{label}] AUC-PR: {auc_pr:.4f}")
    log.info("\n" + classification_report(y_te, y_pred, target_names=["NO_WIN", "WIN"], digits=3))

    bundle = {
        "model":           pipe,
        "feature_columns": features,
        "threshold":       0.50,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       int(len(X_tr)),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "layer":           label,
    }
    with open(pkl_s1, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"✅ {pkl_s1}  ({pkl_s1.stat().st_size / 1024:.0f} KB)")

    if pkl_s2 is not None:
        wins_tr = train_df[train_df["outcome_label"].isin(["WIN_40", "WIN_20"])].copy()
        wins_tr["target_s2"] = (wins_tr["outcome_label"] == "WIN_40").astype(int)
        if len(wins_tr) >= 30:
            pipe2 = _build_pipeline(algo)
            pipe2.fit(wins_tr[features].values.astype(np.float32), wins_tr["target_s2"].values)
            bundle2 = {
                "model":           pipe2,
                "feature_columns": features,
                "threshold":       0.55,
                "algorithm":       algo,
                "n_samples":       len(wins_tr),
                "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
                "layer":           label,
            }
            with open(pkl_s2, "wb") as f:
                pickle.dump(bundle2, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"✅ {pkl_s2}  ({pkl_s2.stat().st_size / 1024:.0f} KB)")
        else:
            log.info(f"[{label}] Stage 2 saltado ({len(wins_tr)} wins < 30)")


# ── Ponto de entrada público (scheduler do bot) ───────────────────────────────

def run_auto() -> None:
    log.info("=" * 55)
    log.info("AUTO RUN — dual-layer ML")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        log.error("scikit-learn não instalado")
        return

    data_dir = Path("/data") if Path("/data").exists() else Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    pkl_price     = data_dir / "dip_model_price.pkl"
    pkl_s1        = data_dir / "dip_model_stage1.pkl"
    pkl_s2        = data_dir / "dip_model_stage2.pkl"
    parquet_price = data_dir / "ml_training_price.parquet"
    parquet_fund  = data_dir / "ml_training_fund.parquet"

    universe = _load_universe()

    start_p, end_p = _window(YEARS_PRICE)
    existing_p = pd.read_parquet(parquet_price) if parquet_price.exists() else pd.DataFrame()
    new_p = backfill_price(start=start_p, end=end_p, tickers=universe, existing_df=existing_p)
    df_p  = load_and_slide(parquet_price, start_p, new_p, skip_exit_on_empty=True)
    _train_layer(df_p, FEATURES_PRICE, pkl_price, None, "rf", "CamadaA")

    start_f, end_f = _window(YEARS_FUND)
    existing_f = pd.read_parquet(parquet_fund) if parquet_fund.exists() else pd.DataFrame()
    new_f = backfill_fund(start=start_f, end=end_f, tickers=universe, existing_df=existing_f)
    df_f  = load_and_slide(parquet_fund, start_f, new_f, skip_exit_on_empty=True)
    _train_layer(df_f, FEATURES_FUND, pkl_s1, pkl_s2, "rf", "CamadaB")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DipRadar — Dual-Layer ML\n"
            "Universo: hardcoded (~830 tickers únicos, zero scraping)\n\n"
            "COLAB (batch incremental, ~200 tickers por sessão de 90 min):\n"
            "  Sessão 1: python bootstrap_ml.py --layer price --slice 0 200 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 2: python bootstrap_ml.py --layer price --slice 200 400 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 3: python bootstrap_ml.py --layer price --slice 400 600 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 4: python bootstrap_ml.py --layer price --slice 600 830 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão final: python bootstrap_ml.py --layer price --skip-backfill --drive-dir /content/drive/MyDrive/DipRadar"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--algo",          choices=["rf", "xgb"], default="rf")
    p.add_argument("--layer",         choices=["all", "price", "fund"], default="all")
    p.add_argument("--years-price",   type=int, default=YEARS_PRICE)
    p.add_argument("--years-fund",    type=int, default=YEARS_FUND)
    p.add_argument("--dip-thresh",    type=float, default=0.04)
    p.add_argument("--skip-backfill", action="store_true",
                   help="Salta o backfill e treina directamente com o Parquet existente")
    p.add_argument("--force-full",    action="store_true",
                   help="Ignora dados existentes e refaz backfill completo")
    p.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"), default=None,
                   help="Limita o backfill a UNIVERSE[START:END]. Exemplo: --slice 0 200")
    p.add_argument("--drive-dir", type=str, default=None, metavar="DIR",
                   help="Directório onde os Parquets e .pkl são lidos/guardados.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import sklearn  # noqa: F401
    except ImportError:
        log.error("scikit-learn não instalado")
        sys.exit(1)

    if args.drive_dir:
        data_dir = Path(args.drive_dir)
    elif Path("/data").exists():
        data_dir = Path("/data")
    else:
        data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    pkl_price     = data_dir / "dip_model_price.pkl"
    pkl_s1        = data_dir / "dip_model_stage1.pkl"
    pkl_s2        = data_dir / "dip_model_stage2.pkl"
    parquet_price = data_dir / "ml_training_price.parquet"
    parquet_fund  = data_dir / "ml_training_fund.parquet"

    universe = _load_universe()
    log.info(f"📋 Universo total: {len(universe)} tickers ML")

    if args.slice:
        s_start, s_end = args.slice
        tickers = universe[s_start:s_end]
        log.info(f"🔪 Slice UNIVERSE[{s_start}:{s_end}] = {len(tickers)} tickers")
    else:
        tickers = universe

    run_price = args.layer in ("all", "price")
    run_fund  = args.layer in ("all", "fund")

    if run_price:
        start_p, end_p = _window(args.years_price)
        log.info(f"[CamadaA] Janela: {start_p} → {end_p} | data_dir: {data_dir}")
        if args.skip_backfill:
            df_p = pd.read_parquet(parquet_price) if parquet_price.exists() else pd.DataFrame()
        else:
            existing_p = pd.DataFrame() if args.force_full else (
                pd.read_parquet(parquet_price) if parquet_price.exists() else pd.DataFrame()
            )
            new_p = backfill_price(
                start=start_p, end=end_p, tickers=tickers,
                dip_thresh=args.dip_thresh, existing_df=existing_p,
            )
            df_p = load_and_slide(parquet_price, start_p, new_p, skip_exit_on_empty=bool(args.slice))
        if not df_p.empty and args.skip_backfill:
            _train_layer(df_p, FEATURES_PRICE, pkl_price, None, args.algo, "CamadaA")
        elif not args.skip_backfill and not args.slice:
            _train_layer(df_p, FEATURES_PRICE, pkl_price, None, args.algo, "CamadaA")
        else:
            n = len(df_p) if not df_p.empty else 0
            log.info(f"[CamadaA] Batch concluído — Parquet acumulado: {n} registos. Treino adiado para sessão final (--skip-backfill).")

    if run_fund:
        start_f, end_f = _window(args.years_fund)
        log.info(f"[CamadaB] Janela: {start_f} → {end_f} | data_dir: {data_dir}")
        if args.skip_backfill:
            df_f = pd.read_parquet(parquet_fund) if parquet_fund.exists() else pd.DataFrame()
        else:
            existing_f = pd.DataFrame() if args.force_full else (
                pd.read_parquet(parquet_fund) if parquet_fund.exists() else pd.DataFrame()
            )
            new_f = backfill_fund(
                start=start_f, end=end_f, tickers=tickers,
                dip_thresh=args.dip_thresh, existing_df=existing_f,
            )
            df_f = load_and_slide(parquet_fund, start_f, new_f, skip_exit_on_empty=bool(args.slice))
        if not df_f.empty and args.skip_backfill:
            _train_layer(df_f, FEATURES_FUND, pkl_s1, pkl_s2, args.algo, "CamadaB")
        elif not args.skip_backfill and not args.slice:
            _train_layer(df_f, FEATURES_FUND, pkl_s1, pkl_s2, args.algo, "CamadaB")
        else:
            n = len(df_f) if not df_f.empty else 0
            log.info(f"[CamadaB] Batch concluído — Parquet acumulado: {n} registos. Treino adiado para sessão final (--skip-backfill).")

    log.info("=" * 55)
    log.info("CONCLUÍDO")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
