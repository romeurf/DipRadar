"""
bootstrap_ml.py — Dual-Layer ML: Camada A (preço, 20 anos) + Camada B (fundamentais, 3 anos).

ARQUITECTURA DE FEATURES — CONTRATO ÚNICO
==========================================
Este ficheiro importa FEATURE_COLUMNS directamente de ml_features.py.
Isso garante que o vector de treino é IDÊNTICO ao vector de inferência.
Nunca duplicar a lista de features aqui — a source of truth é ml_features.py.

MACRO — BULK FETCH (3 pedidos de rede para 20 anos de história)
================================================================
Antes de iterar sobre tickers, descarregamos as séries temporais completas:
  - ^VIX  (yfinance)
  - SPY   (yfinance — para spy_drawdown_5d e macro_score)
  - T10Y2Y (FRED via pandas_datareader — yield spread 10Y-2Y)

Juntamos num DataFrame indexado por data com ffill() para cobrir feriados/fins-de-semana.
Dentro do backfill, cada alerta faz um lookup O(1): macro_row = global_macro_df.loc[date].
Desta forma nunca há rate-limiting: 3 pedidos totais para qualquer número de alertas.

UNIVERSO HARDCODED (~880 tickers únicos, zero scraping)
=========================================================
S&P 500 + Nasdaq 100 + STOXX 200 + FTSE 100 + carteira + watchlist.

MODO AUTOMÁTICO (agendado pelo bot, corre às 02:00 UTC todos os dias):
    python bootstrap_ml.py

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

OUTPUT:
    <data_dir>/dip_model_price.pkl    ← Camada A
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

# ── CONTRATO ÚNICO DE FEATURES ────────────────────────────────────────────────
# Importamos directamente de ml_features para garantir que treino == inferência.
# Se adicionares uma feature em ml_features.py, ela aparece automaticamente aqui.
from ml_features import FEATURE_COLUMNS, N_FEATURES

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bootstrap_ml] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bootstrap_ml")

YEARS_PRICE = 20
YEARS_FUND  = 3

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
    "SAP.DE","SIE.DE","ALV.DE","MRK.DE","BMW.DE","MBG.DE","BAS.DE","BAYN.DE","DTE.DE",
    "DBK.DE","VOW3.DE","ADS.DE","HNR1.DE","RWE.DE","HEI.DE","ENR.DE","EOAN.DE","CON.DE",
    "FRE.DE","ZAL.DE","LIN.DE","MUV2.DE","DHER.DE","SHL.DE","MTX.DE","1COV.DE","BOSS.DE",
    "HAB.DE","HFG.DE","VNA.DE","BEI.DE","SYMB.DE","SY1.DE","KGX.DE","AIXA.DE","CARL.DE",
    "QIA.DE","EVD.DE","DHL.DE","PUM.DE",
    "MC.PA","OR.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","ACA.PA","SGO.PA","SU.PA",
    "DG.PA","RI.PA","CS.PA","KER.PA","ATO.PA","CAP.PA","VIE.PA","LR.PA","DSY.PA",
    "HO.PA","ML.PA","ORA.PA","ENGI.PA","STM.PA","SAF.PA","BN.PA","SG.PA","EL.PA",
    "RMS.PA","WLN.PA","CA.PA","TEP.PA","RNO.PA","URW.PA","PUB.PA",
    "ASML.AS","HEIA.AS","PHIA.AS","REN.AS","UNA.AS","INGA.AS","ABN.AS","NN.AS",
    "RAND.AS","AKZA.AS","IMCD.AS","WKL.AS","ADY.AS","BESI.AS","LIGHT.AS",
    "NESN.SW","NOVN.SW","ROG.SW","UBSG.SW","CSGN.SW","ABBN.SW","ZURN.SW","GIVN.SW",
    "LONN.SW","CFR.SW","SREN.SW","SCMN.SW","BAER.SW","ALC.SW","GEBN.SW","SIKA.SW",
    "ITX.MC","SAN.MC","TEF.MC","BBVA.MC","IBE.MC","AMS.MC","REP.MC","CABK.MC",
    "BKT.MC","GRF.MC","MAP.MC","ENG.MC","NTGY.MC","MTS.MC",
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","STM.MI","LDO.MI","MB.MI","FCA.MI",
    "PRY.MI","RACE.MI","MONC.MI","BMED.MI","PST.MI",
    "VOLV-B.ST","ERIC-B.ST","ATCO-A.ST","SEB-A.ST","SHB-A.ST","SWED-A.ST","INVE-B.ST",
    "SSAB-A.ST","SAND.ST","SKF-B.ST","HM-B.ST","ALFA.ST","NIBE-B.ST","TELIA.ST",
    "NOVO-B.CO","ORSTED.CO","CARL-B.CO","DSV.CO","COLO-B.CO","DEMANT.CO","GN.CO",
    "EQNR.OL","DNB.OL","MOWI.OL","TEL.OL","YAR.OL","NHY.OL","ORKLA.OL",
    "NOKIA.HE","FORTUM.HE","NESTE.HE","KNEBV.HE","STERV.HE",
    "UCB.BR","ABI.BR","SOLB.BR","AGS.BR","GLPG.BR",
    "CRH.I","KYGA.I","AIB.I","BIRG.I",
    "EDP.LS","GALP.LS","BCP.LS","EDPR.LS",
    "VIG.VI","OMV.VI","EBS.VI",
    "PKN.WA","PKO.WA","PZU.WA","DNP.WA","LPP.WA",
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
    "BME.L","CRDA.L","ENT.L","FLTR.L","KGF.L","MNG.L","SN.L","WEIR.L","WTB.L","XP.L",
]

_CARTEIRA = [
    "NVO","ADBE","UBER","MSFT","PINS","ADP","CRM","VICI","CRWD","PLTR","NOW","DUOL","EUNL.DE",
]

_WATCHLIST = [
    "O","MDT","ABBV","LMT","RTX","PANW","TSM","AVGO","ALV.DE","IEMA",
]


def _load_universe() -> list[str]:
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
    log.info(f"[universe] {len(all_tickers)} tickers únicos (hardcoded, zero scraping)")
    return all_tickers


def _window(years: int) -> tuple[date, date]:
    end   = date.today() - timedelta(days=1)
    start = end.replace(year=end.year - years)
    return start, end


def _normalize_history_index(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return hist
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        hist.index = idx.tz_localize(None)
    return hist.sort_index()


# ── MACRO BULK FETCH ──────────────────────────────────────────────────────────
# Descarga uma única vez as séries macro para toda a janela histórica.
# Resultado: DataFrame indexado por date com colunas:
#   vix, spy_close, spy_ret_1d, t10y2y
# + colunas derivadas: macro_score, spy_drawdown_5d
#
# 3 pedidos de rede no total. Lookup O(1) por alert_date no backfill.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_macro_score(vix: float, spy_ret_5d: float, t10y2y: float) -> int:
    """
    Replica a lógica do macro_semaphore.py para dados históricos.
    Retorna int 0–4: BEAR=0-1, NEUTRAL=2, BULL=3-4.
    """
    score = 2  # neutral base

    # VIX: medo elevado → bearish
    if vix >= 35:
        score -= 2
    elif vix >= 25:
        score -= 1
    elif vix <= 15:
        score += 1

    # SPY momentum 5d
    if spy_ret_5d <= -3.0:
        score -= 1
    elif spy_ret_5d >= 2.0:
        score += 1

    # Yield spread: invertida é sinal bearish histórico
    if t10y2y <= -0.5:
        score -= 1
    elif t10y2y >= 1.0:
        score += 1

    return int(max(0, min(4, score)))


def build_global_macro_df(start: date, end: date) -> pd.DataFrame:
    """
    Descarrega VIX, SPY e T10Y2Y (FRED) de uma só vez.
    Retorna DataFrame diário com ffill para weekends/feriados, indexado por date.

    Colunas produzidas (alinhadas com ml_features.FEATURE_COLUMNS):
        vix, spy_close, spy_drawdown_5d, t10y2y, macro_score
    """
    import yfinance as yf

    start_str = start.isoformat()
    end_str   = (end + timedelta(days=5)).isoformat()  # margem para ffill

    log.info(f"[macro] Bulk-fetch VIX + SPY ({start_str} → {end_str}) ...")

    # ── 1. VIX ───────────────────────────────────────────────────────
    vix_hist = yf.Ticker("^VIX").history(start=start_str, end=end_str, interval="1d")
    vix_hist = _normalize_history_index(vix_hist)
    vix_s    = vix_hist["Close"].rename("vix")

    # ── 2. SPY ───────────────────────────────────────────────────────
    spy_hist  = yf.Ticker("SPY").history(start=start_str, end=end_str, interval="1d")
    spy_hist  = _normalize_history_index(spy_hist)
    spy_close = spy_hist["Close"].rename("spy_close")

    # spy_drawdown_5d: retorno dos últimos 5 dias de trading (em %)
    spy_ret5d = spy_hist["Close"].pct_change(5) * 100
    spy_ret5d.name = "spy_drawdown_5d"

    # spy_ret_1d: para uso interno no macro_score
    spy_ret1d = spy_hist["Close"].pct_change() * 100
    spy_ret1d.name = "spy_ret_1d"

    # ── 3. T10Y2Y via FRED (pandas_datareader) ───────────────────────
    t10y2y_s: pd.Series
    try:
        from pandas_datareader import data as pdr
        t10y2y_raw = pdr.DataReader("T10Y2Y", "fred", start_str, end_str)
        t10y2y_s   = t10y2y_raw["T10Y2Y"].rename("t10y2y")
        log.info("[macro] T10Y2Y carregado via FRED (pandas_datareader)")
    except Exception as e:
        log.warning(f"[macro] FRED indisponível ({e}) — T10Y2Y = 0.5 (fallback)")
        t10y2y_s = pd.Series(0.5, index=spy_close.index, name="t10y2y")

    # ── Juntar tudo e forward-fill ────────────────────────────────────
    df = pd.concat([vix_s, spy_close, spy_ret5d, spy_ret1d, t10y2y_s], axis=1)
    df.index = pd.to_datetime(df.index).normalize()  # garantir só datas
    df = df.sort_index()

    # Criar índice contínuo diário e ffill para weekends/feriados
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_idx).ffill().bfill()
    df.index = df.index.date  # converter para date puro para lookups simples

    # ── Macro score derivado ──────────────────────────────────────────
    df["macro_score"] = df.apply(
        lambda r: _compute_macro_score(
            vix      = float(r["vix"])            if pd.notna(r["vix"])            else 20.0,
            spy_ret_5d = float(r["spy_drawdown_5d"]) if pd.notna(r["spy_drawdown_5d"]) else 0.0,
            t10y2y   = float(r["t10y2y"])         if pd.notna(r["t10y2y"])         else 0.5,
        ),
        axis=1,
    )

    # Valores de fallback para células ainda em NaN
    df["vix"]             = df["vix"].fillna(20.0)
    df["spy_drawdown_5d"] = df["spy_drawdown_5d"].fillna(0.0)
    df["t10y2y"]          = df["t10y2y"].fillna(0.5)
    df["macro_score"]     = df["macro_score"].fillna(2).astype(int)

    log.info(f"[macro] DataFrame macro pronto: {len(df)} dias | {df.index.min()} → {df.index.max()}")
    return df


def _macro_lookup(macro_df: pd.DataFrame, alert_date: date) -> dict:
    """
    Lookup O(1) no DataFrame global.
    Retorna os 4 campos macro que entram no FEATURE_COLUMNS:
        macro_score, vix, spy_drawdown_5d, sector_drawdown_5d
    sector_drawdown_5d usa spy_drawdown_5d como proxy conservador no histórico
    (sem série histórica por sector ETF sem custo).
    """
    row = macro_df.loc[alert_date] if alert_date in macro_df.index else None
    if row is None:
        # tentar o dia mais próximo (máx ±3 dias)
        for delta in range(1, 4):
            for sign in (1, -1):
                candidate = alert_date + timedelta(days=delta * sign)
                if candidate in macro_df.index:
                    row = macro_df.loc[candidate]
                    break
            if row is not None:
                break

    if row is None:
        return {"macro_score": 2, "vix": 20.0, "spy_drawdown_5d": 0.0, "sector_drawdown_5d": 0.0}

    return {
        "macro_score":        int(row["macro_score"]),
        "vix":                round(float(row["vix"]), 2),
        "spy_drawdown_5d":    round(float(row["spy_drawdown_5d"]), 3),
        # Proxy histórico: sem série por sector ETF; usamos SPY como conservador
        # Em produção o macro_semaphore.py usa o ETF de sector real.
        "sector_drawdown_5d": round(float(row["spy_drawdown_5d"]), 3),
    }


# ── Indicadores técnicos ───────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def calc_atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low   = hist["High"] - hist["Low"]
    high_close = (hist["High"] - hist["Close"].shift()).abs()
    low_close  = (hist["Low"]  - hist["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_volume_ratio(hist: pd.DataFrame, period: int = 20) -> pd.Series:
    avg = hist["Volume"].rolling(period).mean()
    return hist["Volume"] / (avg + 1e-9)


def safe_float(val, default=None):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def outcome_label_alpha(stock_ret: float, spy_ret: float) -> str:
    alpha = stock_ret - spy_ret
    if   alpha >= 30:  return "WIN_40"
    elif alpha >= 15:  return "WIN_20"
    elif alpha >= -10: return "NEUTRAL"
    else:              return "LOSS_15"


def outcome_label(ret: float) -> str:
    if   ret >= 40:  return "WIN_40"
    elif ret >= 20:  return "WIN_20"
    elif ret >= -15: return "NEUTRAL"
    else:            return "LOSS_15"


def label_win_binary(outcome: str) -> int:
    """
    Converte outcome_label para label_win binário (alinhado com ml_features.LABEL_COLUMNS).
    1 = WIN_20 ou WIN_40 (recovery ≥15-20%)
    0 = NEUTRAL ou LOSS
    """
    return 1 if outcome in ("WIN_40", "WIN_20") else 0


def get_price_near(hist: pd.DataFrame, target: date) -> float | None:
    for d in range(-3, 6):
        check = target + timedelta(days=d)
        m = hist[hist.index.date == check]
        if not m.empty:
            return float(m["Close"].iloc[0])
    return None


# ── Point-in-Time Fundamentals ────────────────────────────────────────────────
#
# REGRA INSTITUCIONAL: O único fallback válido para o passado desconhecido é NaN.
# NUNCA usar tk.info como fallback para dados históricos — isso reintroduz
# Look-Ahead Bias com potencialmente anos de informação futura.
#
# tk.info é APENAS válido para:
#   1. Dados estáticos independentes do tempo (sector, industry)
#   2. Vector de inferência de hoje em produção (Point-in-Time válido)
# ─────────────────────────────────────────────────────────────────────────────

def _get_historical_fundamentals(tk, alert_date) -> dict:
    """
    Extrai fundamentais respeitando rigorosamente o Point-in-Time (SEC reporting lag).

    Usa quarterly_income_stmt / quarterly_balance_sheet / quarterly_cashflow
    com lag de 45 dias (tempo mínimo entre fim do trimestre e publicação SEC).

    Retorna NaN para qualquer métrica que o yfinance não consiga fornecer para
    aquele período — o modelo lida com NaNs via SimpleImputer(strategy='median').

    NUNCA faz fallback para tk.info: isso injectaria dados futuros na linha de treino.
    """
    valid_threshold = pd.to_datetime(alert_date) - pd.Timedelta(days=45)

    result = {
        "fcf_yield":      np.nan,
        "revenue_growth": np.nan,
        "gross_margin":   np.nan,
        "de_ratio":       np.nan,
    }

    try:
        inc   = tk.quarterly_income_stmt
        bal   = tk.quarterly_balance_sheet
        cf    = tk.quarterly_cashflow

        # Se a API não devolver nada, aborta em segurança com NaNs
        if inc.empty or bal.empty or cf.empty:
            return result

        # Normalizar colunas para Timestamp (yfinance devolve datas como nomes de colunas)
        def _valid_cols(df: pd.DataFrame) -> list:
            cols = []
            for c in df.columns:
                try:
                    if pd.to_datetime(c) <= valid_threshold:
                        cols.append(c)
                except Exception:
                    pass
            return cols

        inc_cols = _valid_cols(inc)
        bal_cols = _valid_cols(bal)
        cf_cols  = _valid_cols(cf)

        # Sem histórico tão antigo: devolvemos NaN — NUNCA tk.info
        if not inc_cols or not bal_cols or not cf_cols:
            return result

        # Trimestre válido mais recente face à data do alerta
        t_inc = max(inc_cols, key=pd.to_datetime)
        t_bal = max(bal_cols, key=pd.to_datetime)
        t_cf  = max(cf_cols,  key=pd.to_datetime)

        # ── Gross Margin ─────────────────────────────────────────────
        gross_profit  = inc[t_inc].get("Gross Profit",  np.nan)
        total_revenue = inc[t_inc].get("Total Revenue", np.nan)
        if pd.notna(gross_profit) and pd.notna(total_revenue) and total_revenue > 0:
            result["gross_margin"] = float(gross_profit) / float(total_revenue)

        # ── Revenue Growth (QoQ: trimestre actual vs mesmo trimestre do ano anterior) ──
        # Ordena colunas inc por data desc e tenta comparar com 4 trimestres atrás
        try:
            all_inc_sorted = sorted(
                [c for c in inc.columns if pd.to_datetime(c) <= valid_threshold],
                key=pd.to_datetime, reverse=True
            )
            if len(all_inc_sorted) >= 5:
                rev_now  = inc[all_inc_sorted[0]].get("Total Revenue", np.nan)
                rev_prev = inc[all_inc_sorted[4]].get("Total Revenue", np.nan)
                if pd.notna(rev_now) and pd.notna(rev_prev) and rev_prev > 0:
                    result["revenue_growth"] = (float(rev_now) - float(rev_prev)) / abs(float(rev_prev))
        except Exception:
            pass

        # ── FCF Yield ────────────────────────────────────────────────
        try:
            op_cf  = cf[t_cf].get("Operating Cash Flow",   np.nan)
            capex  = cf[t_cf].get("Capital Expenditure",   np.nan)
            # capex no yfinance é negativo por convenção
            if pd.notna(op_cf) and pd.notna(capex):
                fcf_ttm = float(op_cf) + float(capex)   # capex já é negativo
                # Market cap histórico: usamos price × shares da bal sheet como proxy
                shares = bal[t_bal].get("Ordinary Shares Number", np.nan)
                # fallback: Common Stock
                if pd.isna(shares):
                    shares = bal[t_bal].get("Common Stock", np.nan)
                # Para calcular market cap histórico aproximado, precisamos do preço
                # nessa data — não está disponível aqui sem hist extra.
                # Guardamos o FCF absoluto; o ratio é calculado abaixo com o preço do alerta.
                result["_fcf_abs"] = fcf_ttm  # chave interna, removida depois
        except Exception:
            pass

        # ── D/E Ratio ────────────────────────────────────────────────
        try:
            total_debt   = bal[t_bal].get("Total Debt",           np.nan)
            stockholder_eq = bal[t_bal].get("Stockholders Equity", np.nan)
            if pd.isna(stockholder_eq):
                stockholder_eq = bal[t_bal].get("Total Equity Gross Minority Interest", np.nan)
            if pd.notna(total_debt) and pd.notna(stockholder_eq) and stockholder_eq != 0:
                result["de_ratio"] = float(total_debt) / abs(float(stockholder_eq)) * 100
        except Exception:
            pass

    except Exception:
        # Erros de rede ou parsing do yfinance: modelo lida com NaNs
        pass

    return result


# ── Backfill — Camada A (preço, 20 anos) ──────────────────────────────────────

def backfill_price(
    start: date,
    end: date,
    tickers: list[str],
    macro_df: pd.DataFrame,
    dip_thresh: float = 0.04,
    max_per_ticker: int = 15,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Gera alertas históricos com o vector de features completo (16 features).
    macro_df é o DataFrame global pre-computado — zero chamadas de rede adicionais.
    """
    import yfinance as yf

    existing_keys: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        existing_keys = set(zip(existing_df["symbol"].astype(str), existing_df["alert_date"].astype(str)))

    start_str = start.isoformat()
    fetch_end = min(end + timedelta(days=200), date.today()).isoformat()
    start_ts  = pd.Timestamp(start)
    end_ts    = pd.Timestamp(end)

    log.info(f"[CamadaA] {start_str} → {end.isoformat()} | {len(tickers)} tickers | existentes: {len(existing_keys)}")
    log.info(f"[CamadaA] Contrato: {N_FEATURES} features (alinhado com ml_features.FEATURE_COLUMNS)")

    spy_close_map: dict[date, float] = {}
    if "spy_close" in macro_df.columns:
        spy_close_map = {d: float(v) for d, v in macro_df["spy_close"].items() if pd.notna(v)}

    all_alerts: list[dict] = []

    for i, ticker in enumerate(tickers):
        try:
            hist = yf.Ticker(ticker).history(start=start_str, end=fetch_end, interval="1d")
            hist = _normalize_history_index(hist)
            if hist.empty or len(hist) < 60:
                continue

            # ── indicadores técnicos ───────────────────────────────────
            hist["rsi"]    = calc_rsi(hist["Close"])
            hist["ret_1d"] = hist["Close"].pct_change() * 100
            hist["atr"]    = calc_atr(hist)
            hist["vol_ratio"] = calc_volume_ratio(hist)

            # Drawdown 52 semanas (= drawdown_52w em ml_features)
            roll_max        = hist["Close"].rolling(252, min_periods=30).max()
            hist["ddp_52w"] = (hist["Close"] - roll_max) / roll_max * 100

            # ATR ratio (= atr_ratio em ml_features)
            hist["atr_ratio"] = hist["atr"] / (hist["Close"] + 1e-9)

            # Beta rolante 252d vs SPY
            spy_ret_aligned = pd.Series(
                {pd.Timestamp(d): (spy_close_map.get(d, np.nan)) for d in spy_close_map}
            ).pct_change() * 100
            spy_aligned = spy_ret_aligned.reindex(hist.index, method="nearest")
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
                hist_after = hist[hist.index.date > alert_date]
                if hist_after.empty:
                    continue

                entry = float(row["Close"])

                # outcomes
                p3m = get_price_near(hist_after, alert_date + timedelta(days=91))
                p6m = get_price_near(hist_after, alert_date + timedelta(days=182))
                if p3m is None and p6m is None:
                    continue

                r3m = (p3m - entry) / entry * 100 if p3m else None
                r6m = (p6m - entry) / entry * 100 if p6m else None

                def spy_ret_period(days: int) -> float | None:
                    target = alert_date + timedelta(days=days)
                    spy_entry = spy_close_map.get(alert_date)
                    if spy_entry is None:
                        for d in range(-3, 6):
                            spy_entry = spy_close_map.get(alert_date + timedelta(days=d))
                            if spy_entry:
                                break
                    if spy_entry is None:
                        return None
                    for d in range(-3, 6):
                        sp = spy_close_map.get(target + timedelta(days=d))
                        if sp is not None:
                            return (sp - spy_entry) / spy_entry * 100
                    return None

                spy_r3m = spy_ret_period(91)
                spy_r6m = spy_ret_period(182)
                ref      = r6m if r6m is not None else r3m
                spy_ref  = (spy_r6m if r6m is not None else spy_r3m) or 0.0
                if ref is None:
                    continue

                # ── Lookup macro O(1) ──────────────────────────────────
                macro = _macro_lookup(macro_df, alert_date)

                # ── Construir vector com FEATURE_COLUMNS exactos ───────
                # Nota: quality_score, fcf_yield, etc. (Stage 1) não estão
                # disponíveis no histórico de preços — usamos fallbacks neutros.
                # A Camada B (backfill_fund) preenche esses campos reais.
                alert_row: dict = {
                    # Stage 0 — Macro (4 features, valores reais históricos)
                    "macro_score":        macro["macro_score"],
                    "vix":                macro["vix"],
                    "spy_drawdown_5d":    macro["spy_drawdown_5d"],
                    "sector_drawdown_5d": macro["sector_drawdown_5d"],

                    # Stage 1 — Quality / Value (fallbacks neutros na Camada A)
                    "fcf_yield":          0.04,
                    "revenue_growth":     0.05,
                    "gross_margin":       0.35,
                    "de_ratio":           80.0,
                    "pe_vs_fair":         1.0,
                    "analyst_upside":     0.10,
                    "quality_score":      0.50,

                    # Stage 2 — Timing (valores reais)
                    "drop_pct_today":     round(float(row["ret_1d"]), 3),
                    "drawdown_52w":       round(safe_float(row["ddp_52w"], -15.0), 3),
                    "rsi_14":             round(float(np.clip(safe_float(row["rsi"], 50.0), 0, 100)), 1),
                    "atr_ratio":          round(safe_float(row["atr_ratio"], 0.02), 6),
                    "volume_spike":       round(safe_float(row["vol_ratio"], 1.0), 4),

                    # Metadados (não entram no modelo)
                    "symbol":             ticker,
                    "alert_date":         alert_date.isoformat(),
                    "price":              round(entry, 2),

                    # Labels
                    "label_win":          label_win_binary(outcome_label_alpha(ref, spy_ref)),
                    "label_further_drop": None,

                    # Labels auxiliares (para análise)
                    "outcome_label":      outcome_label_alpha(ref, spy_ref),
                    "return_3m":          round(r3m, 2) if r3m is not None else None,
                    "return_6m":          round(r6m, 2) if r6m is not None else None,
                    "spy_return_ref":     round(spy_ref, 2),
                }

                # Verificar contrato: todos os FEATURE_COLUMNS presentes
                missing = [c for c in FEATURE_COLUMNS if c not in alert_row]
                if missing:
                    log.warning(f"[CamadaA] {ticker} {alert_date}: features em falta {missing} — alerta ignorado")
                    continue

                all_alerts.append(alert_row)

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(tickers)}] {len(all_alerts)} alertas")
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  ERRO {ticker}: {e}")

    log.info(f"[CamadaA] {len(all_alerts)} novos alertas com {N_FEATURES} features")
    return pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame()


# ── Backfill — Camada B (fundamentais, 7 anos) ────────────────────────────────

def backfill_fund(
    start: date,
    end: date,
    tickers: list[str],
    macro_df: pd.DataFrame,
    dip_thresh: float = 0.04,
    max_per_ticker: int = 8,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Camada B: enriquece com fundamentais Point-in-Time reais + macro histórico.

    Preenche os campos Stage 1 (fcf_yield, revenue_growth, gross_margin, de_ratio)
    com _get_historical_fundamentals() — rigorosamente Point-in-Time, lag SEC 45 dias.

    NUNCA usa tk.info como fallback para dados históricos (Look-Ahead Bias).
    tk.info é usado APENAS para dados estáticos (sector) e para quality_score/
    pe_vs_fair que não têm série histórica gratuita (são marcados como aproximações).
    """
    import yfinance as yf

    existing_keys: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        existing_keys = set(zip(existing_df["symbol"].astype(str), existing_df["alert_date"].astype(str)))

    start_str = start.isoformat()
    fetch_end = min(end + timedelta(days=200), date.today()).isoformat()
    start_ts  = pd.Timestamp(start)
    end_ts    = pd.Timestamp(end)

    log.info(f"[CamadaB] {start_str} → {end.isoformat()} | {len(tickers)} tickers | existentes: {len(existing_keys)}")

    spy_close_map: dict[date, float] = {}
    if "spy_close" in macro_df.columns:
        spy_close_map = {d: float(v) for d, v in macro_df["spy_close"].items() if pd.notna(v)}

    all_alerts: list[dict] = []

    EU_SUFFIXES = {
        ".DE", ".PA", ".L", ".AS", ".SW", ".MC", ".MI",
        ".ST", ".CO", ".OL", ".HE", ".BR", ".I", ".LS",
        ".VI", ".WA",
    }

    for i, ticker in enumerate(tickers):
        try:
            suffix = ("." + ticker.split(".")[-1]) if "." in ticker else ""
            if suffix and suffix not in EU_SUFFIXES:
                log.debug(f"[CamadaB] {ticker} sufixo '{suffix}' desconhecido — skip")
                continue
            tk   = yf.Ticker(ticker)
            hist = tk.history(start=start_str, end=fetch_end, interval="1d")
            hist = _normalize_history_index(hist)
            if hist.empty or len(hist) < 60:
                continue

            # ── Dados estáticos de tk.info (independentes do tempo) ───
            # ÚNICO uso válido de tk.info na Camada B: sector e quality proxies
            # que não têm série histórica gratuita.
            info = tk.info or {}
            sector_str = info.get("sector", "Unknown") or "Unknown"

            # pe_vs_fair e analyst_upside: sem série histórica no yfinance gratuito.
            # Usamos tk.info actual mas marcamos como "aproximação estática".
            # O modelo deve tratar estes campos com menor peso (sem PIT garantido).
            pe_raw = safe_float(info.get("trailingPE") or info.get("forwardPE"))
            _SECTOR_FAIR_PE = {
                "Technology": 35.0, "Healthcare": 22.0, "Communication Services": 22.0,
                "Financial Services": 13.0, "Financials": 13.0, "Consumer Cyclical": 20.0,
                "Consumer Defensive": 22.0, "Industrials": 20.0, "Energy": 12.0,
                "Utilities": 18.0, "Real Estate": 40.0, "Basic Materials": 14.0, "Materials": 14.0,
            }
            fair_pe    = _SECTOR_FAIR_PE.get(sector_str, 22.0)
            pe_vs_fair = round(pe_raw / fair_pe, 4) if pe_raw and pe_raw > 0 else 1.0

            tgt = safe_float(info.get("targetMeanPrice"))
            cur = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"), 1.0)
            analyst_upside = ((tgt - cur) / cur) if tgt and cur and cur > 0 else 0.10

            mcap = safe_float(info.get("marketCap"), 0) / 1e9

            hist["rsi"]       = calc_rsi(hist["Close"])
            hist["ret_1d"]    = hist["Close"].pct_change() * 100
            hist["atr"]       = calc_atr(hist)
            hist["vol_ratio"] = calc_volume_ratio(hist)
            roll_max          = hist["Close"].rolling(252, min_periods=30).max()
            hist["ddp_52w"]   = (hist["Close"] - roll_max) / roll_max * 100
            hist["atr_ratio"] = hist["atr"] / (hist["Close"] + 1e-9)

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
                hist_after = hist[hist.index.date > alert_date]
                if hist_after.empty:
                    continue

                entry = float(row["Close"])

                # ── Fundamentais Point-in-Time — NÚCLEO ANTI-LOOKAHEAD ──
                # _get_historical_fundamentals respeita o lag SEC de 45 dias.
                # Devolve NaN se o yfinance não tiver história suficiente.
                # NUNCA usa tk.info como fallback — isso seria Look-Ahead Bias.
                fund = _get_historical_fundamentals(tk, alert_date)

                # FCF Yield: combinar FCF absoluto (PIT) com preço de entrada
                fcf_yield_pit = np.nan
                if pd.notna(fund.get("_fcf_abs")) and entry > 0:
                    # Aproximação: FCF trimestral anualizado / market cap naquela data
                    # market cap histórico = preço × shares (não temos shares históricas
                    # sem custo, por isso usamos mcap actual como proxy conservador)
                    mc_actual = safe_float(info.get("marketCap"))
                    if mc_actual and mc_actual > 0:
                        fcf_annualized = float(fund["_fcf_abs"]) * 4
                        fcf_yield_pit = fcf_annualized / mc_actual
                fund.pop("_fcf_abs", None)  # remover chave interna

                # quality_score proxy (derivado de métricas PIT onde disponível)
                gm_pit  = fund.get("gross_margin",   np.nan)
                de_pit  = fund.get("de_ratio",        np.nan)
                revg_pit = fund.get("revenue_growth", np.nan)
                qs = 0.5
                if pd.notna(fcf_yield_pit) and fcf_yield_pit > 0.06: qs += 0.15
                if pd.notna(gm_pit)  and gm_pit  > 0.40: qs += 0.10
                if pd.notna(de_pit)  and de_pit   < 50:  qs += 0.10
                if pd.notna(revg_pit) and revg_pit > 0.10: qs += 0.10
                if analyst_upside > 0.20: qs += 0.05
                quality_score = round(min(qs, 1.0), 4)

                p3m = get_price_near(hist_after, alert_date + timedelta(days=91))
                p6m = get_price_near(hist_after, alert_date + timedelta(days=182))
                if p3m is None and p6m is None:
                    continue

                r3m = (p3m - entry) / entry * 100 if p3m else None
                r6m = (p6m - entry) / entry * 100 if p6m else None
                ref = r6m if r6m is not None else r3m
                if ref is None:
                    continue

                # ── Lookup macro O(1) ──────────────────────────────────
                macro = _macro_lookup(macro_df, alert_date)

                alert_row: dict = {
                    # Stage 0 — Macro
                    "macro_score":        macro["macro_score"],
                    "vix":                macro["vix"],
                    "spy_drawdown_5d":    macro["spy_drawdown_5d"],
                    "sector_drawdown_5d": macro["sector_drawdown_5d"],

                    # Stage 1 — Quality / Value (Point-in-Time via quarterly statements)
                    # NaN onde o yfinance não tem história — o modelo lida via imputer
                    "fcf_yield":          round(float(fcf_yield_pit), 6) if pd.notna(fcf_yield_pit) else np.nan,
                    "revenue_growth":     round(float(fund["revenue_growth"]), 4) if pd.notna(fund["revenue_growth"]) else np.nan,
                    "gross_margin":       round(float(fund["gross_margin"]), 4) if pd.notna(fund["gross_margin"]) else np.nan,
                    "de_ratio":           round(float(fund["de_ratio"]), 2) if pd.notna(fund["de_ratio"]) else np.nan,
                    # pe_vs_fair e analyst_upside: aproximação estática (tk.info actual)
                    # sem série histórica gratuita — menor fiabilidade PIT
                    "pe_vs_fair":         round(float(pe_vs_fair), 4),
                    "analyst_upside":     round(float(analyst_upside), 4),
                    "quality_score":      quality_score,

                    # Stage 2 — Timing
                    "drop_pct_today":     round(float(row["ret_1d"]), 3),
                    "drawdown_52w":       round(safe_float(row["ddp_52w"], -15.0), 3),
                    "rsi_14":             round(float(np.clip(safe_float(row["rsi"], 50.0), 0, 100)), 1),
                    "atr_ratio":          round(safe_float(row["atr_ratio"], 0.02), 6),
                    "volume_spike":       round(safe_float(row["vol_ratio"], 1.0), 4),

                    # Metadados
                    "symbol":             ticker,
                    "alert_date":         alert_date.isoformat(),
                    "price":              round(entry, 2),
                    "sector":             sector_str,
                    "market_cap_b":       round(float(mcap), 2),

                    # Labels
                    "label_win":          label_win_binary(outcome_label(ref)),
                    "label_further_drop": None,
                    "outcome_label":      outcome_label(ref),
                    "return_3m":          round(r3m, 2) if r3m is not None else None,
                    "return_6m":          round(r6m, 2) if r6m is not None else None,
                }

                missing = [c for c in FEATURE_COLUMNS if c not in alert_row]
                if missing:
                    log.warning(f"[CamadaB] {ticker} {alert_date}: features em falta {missing} — alerta ignorado")
                    continue

                all_alerts.append(alert_row)

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(tickers)}] {len(all_alerts)} alertas")
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  ERRO {ticker}: {e}")

    log.info(f"[CamadaB] {len(all_alerts)} novos alertas com {N_FEATURES} features")
    return pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame()


# ── Janela deslizante ──────────────────────────────────────────────────────────

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

    combined = pd.concat([existing, new_df], ignore_index=True) if not new_df.empty else existing
    combined["alert_date"] = combined["alert_date"].astype(str)
    combined.drop_duplicates(subset=["symbol", "alert_date"], keep="last", inplace=True)
    combined.sort_values("alert_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.to_parquet(parquet, index=False)
    log.info(f"📦 Parquet: {len(combined)} registos → {parquet}")
    return combined


# ── Pipeline de treino ─────────────────────────────────────────────────────────

def _build_pipeline(algo: str = "rf"):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=5,
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
    pkl_s1: Path,
    pkl_s2: Path | None,
    algo: str,
    label: str,
) -> None:
    """
    Treina com FEATURE_COLUMNS importado de ml_features.py.
    Usa label_win (binário) como target — alinhado com ml_features.LABEL_COLUMNS.
    SimpleImputer(strategy='median') lida com NaNs de fundamentais PIT em falta.
    """
    from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve

    if df.empty:
        log.warning(f"[{label}] DataFrame vazio — treino saltado.")
        return

    # Validar contrato de features
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        log.error(f"[{label}] Features em falta no Parquet: {missing_features}")
        log.error(f"[{label}] Certifica-te de que o backfill foi gerado com esta versão do bootstrap_ml.py")
        return

    df2 = df[df["label_win"].notna()].copy()
    df2["label_win"] = df2["label_win"].astype(int)

    if len(df2) < 30 or df2["label_win"].sum() < 10:
        log.error(f"[{label}] Dados insuficientes: {len(df2)} linhas, {int(df2['label_win'].sum())} wins")
        return

    df2 = df2.sort_values("alert_date").reset_index(drop=True)
    split    = int(len(df2) * 0.80)
    train_df = df2.iloc[:split]
    test_df  = df2.iloc[split:]

    X_tr = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_tr = train_df["label_win"].values
    X_te = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_te = test_df["label_win"].values

    log.info(f"[{label}] {algo.upper()} | {N_FEATURES} features | train={len(X_tr)} test={len(X_te)} wins={y_tr.sum()}")

    pipe = _build_pipeline(algo)
    if algo == "xgb":
        ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
        pipe.named_steps["clf"].set_params(scale_pos_weight=ratio)
    pipe.fit(X_tr, y_tr)

    probs  = pipe.predict_proba(X_te)[:, 1]
    auc_pr = average_precision_score(y_te, probs)

    precisions, recalls, thresholds = precision_recall_curve(y_te, probs)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_thresh = float(thresholds[np.argmax(f1s)])
    best_thresh = round(max(0.30, min(best_thresh, 0.70)), 3)
    log.info(f"[{label}] Threshold óptimo F1: {best_thresh:.3f}")

    y_pred = (probs >= best_thresh).astype(int)
    log.info(f"[{label}] AUC-PR: {auc_pr:.4f}")
    log.info("\n" + classification_report(y_te, y_pred, target_names=["NO_WIN", "WIN"], digits=3))

    bundle = {
        "model":           pipe,
        "feature_columns": FEATURE_COLUMNS,   # contrato explícito no pkl
        "n_features":      N_FEATURES,
        "threshold":       best_thresh,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       int(len(X_tr)),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "layer":           label,
    }
    with open(pkl_s1, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"✅ {pkl_s1}  ({pkl_s1.stat().st_size / 1024:.0f} KB)")

    # Stage 2: dentro dos wins, distinguir WIN_40 vs WIN_20
    if pkl_s2 is not None and "outcome_label" in df2.columns:
        wins_tr = train_df[train_df["label_win"] == 1].copy()
        wins_tr["target_s2"] = (wins_tr["outcome_label"] == "WIN_40").astype(int)
        if len(wins_tr) >= 30:
            pipe2 = _build_pipeline(algo)
            pipe2.fit(wins_tr[FEATURE_COLUMNS].values.astype(np.float32), wins_tr["target_s2"].values)
            bundle2 = {
                "model":           pipe2,
                "feature_columns": FEATURE_COLUMNS,
                "n_features":      N_FEATURES,
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


# ── Ponto de entrada público (scheduler do bot) ────────────────────────────────

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

    # Macro bulk fetch — 3 pedidos de rede para toda a história
    start_p, end_p = _window(YEARS_PRICE)
    macro_df = build_global_macro_df(start_p, end_p)

    existing_p = pd.read_parquet(parquet_price) if parquet_price.exists() else pd.DataFrame()
    new_p = backfill_price(start=start_p, end=end_p, tickers=universe, macro_df=macro_df, existing_df=existing_p)
    df_p  = load_and_slide(parquet_price, start_p, new_p, skip_exit_on_empty=True)
    _train_layer(df_p, pkl_price, None, "rf", "CamadaA")

    start_f, end_f = _window(YEARS_FUND)
    # Reutiliza macro_df (já cobre os 3 anos também — a janela de 20a é mais larga)
    existing_f = pd.read_parquet(parquet_fund) if parquet_fund.exists() else pd.DataFrame()
    new_f = backfill_fund(start=start_f, end=end_f, tickers=universe, macro_df=macro_df, existing_df=existing_f)
    df_f  = load_and_slide(parquet_fund, start_f, new_f, skip_exit_on_empty=True)
    _train_layer(df_f, pkl_s1, pkl_s2, "rf", "CamadaB")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DipRadar — Dual-Layer ML (Training-Serving Skew corrigido)\n"
            "Contrato de features: importado de ml_features.FEATURE_COLUMNS\n\n"
            "COLAB (batch incremental, ~200 tickers por sessão de 90 min):\n"
            "  Sessão 1: python bootstrap_ml.py --layer price --slice 0 200 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 2: python bootstrap_ml.py --layer price --slice 200 400 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 3: python bootstrap_ml.py --layer price --slice 400 600 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 4: python bootstrap_ml.py --layer price --slice 600 800 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão 5: python bootstrap_ml.py --layer price --slice 800 999 --drive-dir /content/drive/MyDrive/DipRadar\n"
            "  Sessão final: python bootstrap_ml.py --layer price --skip-backfill --drive-dir /content/drive/MyDrive/DipRadar"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--algo",           choices=["rf", "xgb"], default="rf")
    p.add_argument("--layer",          choices=["all", "price", "fund"], default="all")
    p.add_argument("--years-price",    type=int, default=YEARS_PRICE)
    p.add_argument("--years-fund",     type=int, default=YEARS_FUND)
    p.add_argument("--dip-thresh",     type=float, default=0.04)
    p.add_argument("--max-per-ticker", type=int, default=15)
    p.add_argument("--skip-backfill",  action="store_true")
    p.add_argument("--force-full",     action="store_true")
    p.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"), default=None)
    p.add_argument("--drive-dir", type=str, default=None, metavar="DIR")
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
    log.info(f"📐 Contrato: {N_FEATURES} features ({FEATURE_COLUMNS})")

    if args.slice:
        s_start, s_end = args.slice
        tickers = universe[s_start:s_end]
        log.info(f"🔪 Slice UNIVERSE[{s_start}:{s_end}] = {len(tickers)} tickers")
    else:
        tickers = universe

    run_price = args.layer in ("all", "price")
    run_fund  = args.layer in ("all", "fund")

    # Macro bulk fetch — feito UMA VEZ independentemente de quantas camadas correm
    start_p, end_p = _window(args.years_price)
    macro_df = build_global_macro_df(start_p, end_p)
    log.info(f"[macro] Bulk fetch concluído — {len(macro_df)} dias de história macro")

    if run_price:
        log.info(f"[CamadaA] Janela: {start_p} → {end_p} | data_dir: {data_dir}")
        if args.skip_backfill:
            df_p = pd.read_parquet(parquet_price) if parquet_price.exists() else pd.DataFrame()
        else:
            existing_p = pd.DataFrame() if args.force_full else (
                pd.read_parquet(parquet_price) if parquet_price.exists() else pd.DataFrame()
            )
            new_p = backfill_price(
                start=start_p, end=end_p, tickers=tickers,
                macro_df=macro_df,
                dip_thresh=args.dip_thresh,
                max_per_ticker=args.max_per_ticker,
                existing_df=existing_p,
            )
            df_p = load_and_slide(parquet_price, start_p, new_p, skip_exit_on_empty=bool(args.slice))

        if not df_p.empty and (args.skip_backfill or not args.slice):
            _train_layer(df_p, pkl_price, None, args.algo, "CamadaA")
        else:
            n = len(df_p) if not df_p.empty else 0
            log.info(f"[CamadaA] Batch concluído — Parquet: {n} registos. Treino adiado para sessão final.")

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
                macro_df=macro_df,
                dip_thresh=args.dip_thresh, existing_df=existing_f,
            )
            df_f = load_and_slide(parquet_fund, start_f, new_f, skip_exit_on_empty=bool(args.slice))

        if not df_f.empty and (args.skip_backfill or not args.slice):
            _train_layer(df_f, pkl_s1, pkl_s2, args.algo, "CamadaB")
        else:
            n = len(df_f) if not df_f.empty else 0
            log.info(f"[CamadaB] Batch concluído — Parquet: {n} registos. Treino adiado para sessão final.")

    log.info("=" * 55)
    log.info("CONCLUÍDO")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
