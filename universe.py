"""
universe.py — Universo global de tickers para o DipRadar 2.0.

Fontes:
  - S&P 500       : Wikipedia (scraped dinamicamente, fallback completo 503)
  - Nasdaq 100    : Wikipedia (scraped dinamicamente, fallback completo 101)
  - STOXX 600     : Lista estática curada (top 200 por liquidez)
  - FTSE 100      : Lista estática curada
  - Carteira atual: Inclui ETFs (EUNL, IEMA) — monitorização de preço apenas

ETFs são excluídos do pipeline ML (sem fundamentais válidos).
Usar ETF_TICKERS para esta exclusão nos outros módulos:

  from universe import get_full_universe, ETF_TICKERS, get_ml_universe
  all_tickers = get_full_universe()   # inclui ETFs
  ml_tickers  = get_ml_universe()    # exclui ETFs
"""

from __future__ import annotations

import logging
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────
# ETFs — monitorizar preço/drawdown MAS excluír do pipeline ML
# Razão: yfinance não devolve FCF, gross margins, D/E, etc. para ETFs.
#         Treinar o modelo com ETFs baralha completamente os features.
# ─────────────────────────────────────────────────────────────────────────
ETF_TICKERS: set[str] = {
    "EUNL",    # iShares Core MSCI World UCITS ETF (carteira)
    "IEMA",    # iShares MSCI EM ESG Leaders (watchlist)
    "IS3N.L",  # alias antigo — manter por compatibilidade com alertas guardados
    # Acrescenta outros ETFs aqui se necessaire
    "IWDA.AS", "CSPX.L", "VUSA.L", "VWRL.L", "VWCE.DE",
    "EMIM.L",  "IQQQ.DE", "XDWD.DE", "DBXD.DE", "EEM",
    "SPY",     "QQQ",     "IVV",    "VTI",    "VOO",
    "GLD",     "SLV",     "TLT",    "HYG",    "LQD",
}

# ─────────────────────────────────────────────────────────────────────────
# Carteira actual do utilizador
# ─────────────────────────────────────────────────────────────────────────
# Inclui ETFs (para monitorização de preço/drawdown via watchlist.py)
# O pipeline ML filtra-os automaticamente com get_ml_universe()
USER_PORTFOLIO: list[str] = [
    # Stocks
    "NVO", "ADBE", "UBER", "MSFT", "PINS",
    "ADP", "CRM", "VICI", "CRWD", "PLTR", "NOW", "DUOL",
    # ETFs (monitorização apenas, sem ML)
    "EUNL",
]

# Watchlist pessoal (tickers de interesse, mesmo fora da carteira)
USER_WATCHLIST: list[str] = [
    "O", "MDT", "ABBV", "LMT", "RTX",
    "PANW", "TSM", "AVGO", "ALV.DE",
    "IEMA",  # ETF — monitorização apenas
]


# ─────────────────────────────────────────────────────────────────────────
# STOXX Europe 600 — top 200 por liquidez/relevância
# ─────────────────────────────────────────────────────────────────────────
_STOXX200: list[str] = [
    # Alemanha
    "SAP.DE", "SIE.DE", "ALV.DE", "MRK.DE", "BMW.DE", "MBG.DE",
    "BAS.DE", "BAYN.DE", "DTE.DE", "DBK.DE", "VOW3.DE", "ADS.DE",
    "HNR1.DE", "RWE.DE", "HEI.DE", "ENR.DE", "EOAN.DE", "CON.DE",
    "FRE.DE", "ZAL.DE", "AIR.DE", "LIN.DE",
    # França
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA",
    "ACA.PA", "SGO.PA", "SU.PA", "DG.PA", "RI.PA", "CS.PA",
    "KER.PA", "ATO.PA", "CAP.PA", "VIE.PA", "LR.PA", "DSY.PA",
    "HO.PA", "ML.PA", "ORA.PA",
    # Holanda
    "ASML.AS", "HEIA.AS", "PHIA.AS", "REN.AS", "UNA.AS",
    "INGA.AS", "ABN.AS", "NN.AS", "RAND.AS", "AKZA.AS",
    # Reino Unido
    "SHEL.L", "AZN.L", "HSBA.L", "BP.L", "RIO.L", "GSK.L",
    "ULVR.L", "DGE.L", "BATS.L", "LLOY.L", "BARC.L", "VOD.L",
    "BT-A.L", "REL.L", "NG.L", "SSE.L", "IMB.L", "EXPN.L",
    "ABF.L", "FERG.L", "CRH.L", "LSEG.L", "NWG.L", "PRU.L",
    # Suíça
    "NESN.SW", "NOVN.SW", "ROG.SW", "UBSG.SW", "CSGN.SW",
    "ABBN.SW", "ZURN.SW", "GIVN.SW", "LONN.SW", "CFR.SW",
    # Espanha
    "ITX.MC", "SAN.MC", "TEF.MC", "BBVA.MC", "IBE.MC",
    "AMS.MC", "REP.MC", "CABK.MC",
    # Itália
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STM.MI",
    "LDO.MI", "MB.MI", "FCA.MI",
    # Suécia
    "VOLV-B.ST", "ERIC-B.ST", "ATCO-A.ST", "SEB-A.ST",
    "SHB-A.ST", "SWED-A.ST", "INVE-B.ST",
    # Dinamarca
    "NOVO-B.CO", "ORSTED.CO", "CARL-B.CO", "DSV.CO",
    # Noruega
    "EQNR.OL", "DNB.OL", "MOWI.OL",
    # Finlândia
    "NOKIA.HE", "FORTUM.HE", "NESTE.HE",
    # Bélgica
    "UCB.BR", "ABI.BR", "SOLB.BR",
    # Irlanda
    "CRH.I", "KYGA.I",
    # Portugal
    "EDP.LS", "GALP.LS", "BCP.LS",
    # Áustria
    "VIG.VI", "OMV.VI",
    # Polónia
    "PKN.WA", "PKO.WA",
]

# ─────────────────────────────────────────────────────────────────────────
# FTSE 100
# ─────────────────────────────────────────────────────────────────────────
_FTSE100: list[str] = [
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L",
    "GSK.L", "DGE.L", "REL.L", "BATS.L", "LLOY.L", "BARC.L",
    "VOD.L", "BT-A.L", "NG.L", "SSE.L", "IMB.L", "CRH.L",
    "LSEG.L", "NWG.L", "PRU.L", "EXPN.L", "ABF.L", "FERG.L",
    "TSCO.L", "CPG.L", "RKT.L", "WPP.L", "IAG.L", "EZJ.L",
    "AAL.L", "ANTO.L", "BHP.L", "GLEN.L", "EVR.L", "MNDI.L",
    "SMDS.L", "SMT.L", "III.L", "LAND.L", "SGRO.L", "HMSO.L",
    "BLND.L", "DLN.L", "PSN.L", "BWY.L", "TW.L", "BA.L",
    "RR.L", "AUTO.L", "OCDO.L", "MKS.L", "NEXT.L",
    "JD.L", "SPX.L", "RS1.L", "DCC.L", "SKG.L", "IMI.L",
    "PHNX.L", "LGEN.L", "AV.L", "ADM.L", "STAN.L",
    "INVP.L", "RTO.L", "SBRY.L", "MRO.L",
]


# ─────────────────────────────────────────────────────────────────────────
# Funções públicas
# ─────────────────────────────────────────────────────────────────────────

def _fetch_sp500() -> list[str]:
    """Scrape S&P 500 da Wikipedia. Fallback completo (503) se falhar."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        logging.info(f"[universe] S&P 500: {len(tickers)} tickers da Wikipedia")
        return tickers
    except Exception as e:
        logging.warning(f"[universe] S&P 500 Wikipedia falhou: {e}. Fallback completo.")
        return _SP500_FALLBACK


def _fetch_nasdaq100() -> list[str]:
    """Scrape Nasdaq 100 da Wikipedia. Fallback completo (101) se falhar."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for t in tables:
            if "Ticker" in t.columns:
                tickers = [str(x) for x in t["Ticker"].tolist()]
                logging.info(f"[universe] Nasdaq 100: {len(tickers)} tickers da Wikipedia")
                return tickers
        raise ValueError("Tabela Nasdaq 100 não encontrada")
    except Exception as e:
        logging.warning(f"[universe] Nasdaq 100 Wikipedia falhou: {e}. Fallback completo.")
        return _NASDAQ100_FALLBACK


@lru_cache(maxsize=1)
def get_full_universe() -> list[str]:
    """
    Universo completo incluindo ETFs.
    Para scan de watchlist/preço. NÃO usar directamente no pipeline ML.
    """
    sp500  = _fetch_sp500()
    ndx100 = _fetch_nasdaq100()

    raw = (
        USER_PORTFOLIO
        + USER_WATCHLIST
        + sp500
        + ndx100
        + _STOXX200
        + _FTSE100
    )

    seen:   set[str]  = set()
    unique: list[str] = []
    for t in raw:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    logging.info(f"[universe] Universo total: {len(unique)} tickers")
    return unique


def get_ml_universe() -> list[str]:
    """
    Universo filtrado para o pipeline ML.
    Remove ETFs — sem fundamentais válidos no yfinance.
    Usar em backtest.py, train_model.py, scan diurno.
    """
    full    = get_full_universe()
    filtered = [t for t in full if t not in ETF_TICKERS]
    excluded = len(full) - len(filtered)
    logging.info(f"[universe] ML universe: {len(filtered)} tickers ({excluded} ETFs excluídos)")
    return filtered


def is_etf(ticker: str) -> bool:
    """Verifica se um ticker é ETF e deve ser excluído do ML."""
    return ticker.upper() in ETF_TICKERS


def get_universe_stats() -> dict:
    """Estatísticas do universo para /admin ou logging."""
    sp500  = _fetch_sp500()
    ndx100 = _fetch_nasdaq100()
    full   = get_full_universe()
    ml     = get_ml_universe()
    return {
        "total":          len(full),
        "ml_eligible":    len(ml),
        "etfs_excluded":  len(full) - len(ml),
        "sp500":          len(sp500),
        "nasdaq100":      len(ndx100),
        "stoxx200":       len(_STOXX200),
        "ftse100":        len(_FTSE100),
        "user_portfolio": len(USER_PORTFOLIO),
        "user_watchlist": len(USER_WATCHLIST),
    }


# ─────────────────────────────────────────────────────────────────────────
# FALLBACKS COMPLETOS — activados se Wikipedia estiver inacessível
# S&P 500: 503 constituintes (Jan 2025)
# Nasdaq 100: 101 constituintes (Jan 2025)
# ─────────────────────────────────────────────────────────────────────────

_SP500_FALLBACK: list[str] = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A",
    "APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL",
    "GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT",
    "AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL",
    "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK",
    "AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX",
    "BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR",
    "BWA","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS",
    "CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT",
    "CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CHRW","CRL",
    "SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO",
    "C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG",
    "COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP",
    "COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA",
    "DAY","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG",
    "DLTR","D","DPZ","DOV","DHI","DTE","DUK","DD","EMN","ETN",
    "EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG",
    "EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
    "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST",
    "FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT",
    "FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
    "GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY",
    "GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE",
    "HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC",
    "ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
    "JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KKR","KLAC","KHC","KR",
    "LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV",
    "LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX",
    "MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
    "MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI",
    "MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE",
    "NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR",
    "NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS",
    "PCAR","PKG","PLTR","PANW","PARA","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG",
    "PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM",
    "QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN",
    "RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI",
    "CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM",
    "SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
    "SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR",
    "TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TPL","TXT",
    "TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN",
    "USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS",
    "VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI",
    "V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR",
    "WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]

_NASDAQ100_FALLBACK: list[str] = [
    "ADBE","ADP","ABNB","ALGN","GOOGL","GOOG","AMZN","AMD","AEP",
    "AMGN","ADI","ANSS","AAPL","AMAT","ASML","AZN","TEAM","ADSK",
    "BIDU","BIIB","BKNG","AVGO","CDNS","CDW","CHTR","CTAS","CSCO",
    "CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX",
    "DDOG","DXCM","FANG","DLTR","EBAY","EA","EXC","FAST","FTNT",
    "GILD","GFS","HON","IDXX","ILMN","INTC","INTU","ISRG","JD",
    "KDP","KLAC","KHC","LRCX","LIN","LULU","MAR","MRVL","MELI",
    "META","MCHP","MU","MSFT","MRNA","MDLZ","MDB","MNST","NFLX",
    "NVDA","NXPI","ORLY","ODFL","ON","PCAR","PANW","PAYX","PYPL",
    "PDD","PEP","QCOM","REGN","ROP","ROST","SBUX","SGEN","SIRI",
    "SNPS","TSLA","TXN","TMUS","TTWO","VRSK","VRTX","WBA","WBD",
    "WDAY","XEL","ZM","ZS",
]
