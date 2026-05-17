"""
universe.py — Universo global de tickers para o DipRadar 2.0.

Fontes:
  - S&P 500       : Wikipedia (scraped dinamicamente, fallback completo 503)
  - Nasdaq 100    : Wikipedia (scraped dinamicamente, fallback completo 101)
  - STOXX 600     : Lista estática curada (top 200 por liquidez)
  - FTSE 100      : Lista estática curada
  - Carteira atual: Inclui ETFs (EUNL.DE, IEMA) — monitorização de preço apenas

ETFs são excluídos do pipeline ML (sem fundamentais válidos).
Usar ETF_TICKERS para esta exclusão nos outros módulos:

  from universe import get_full_universe, ETF_TICKERS, get_ml_universe
  all_tickers = get_full_universe()   # inclui ETFs
  ml_tickers  = get_ml_universe()    # exclui ETFs
"""

from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

# Cache em disco — persiste entre reinícios do Railway
_CACHE_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_FILE = _CACHE_DIR / "ml_universe_cache.json"
# FIX: aumentado de 3 para 7 dias — evita re-scraping frequente que falha
# no Railway (timeouts de rede às 22h30). O universo muda raramente.
_CACHE_DAYS = 7

# ─────────────────────────────────────────────────────────────────────────
# ETFs — monitorizar preço/drawdown MAS excluír do pipeline ML
# Razão: yfinance não devolve FCF, gross margins, D/E, etc. para ETFs.
#         Treinar o modelo com ETFs baralha completamente os features.
# ─────────────────────────────────────────────────────────────────────────
ETF_TICKERS: set[str] = {
    "EUNL.DE",  # iShares Core MSCI World UCITS ETF (carteira)
    "IEMA",    # iShares MSCI EM ESG Leaders (watchlist)
    "IS3N.L",  # alias antigo
    "IWDA.AS", "CSPX.L", "VUSA.L", "VWRL.L", "VWCE.DE",
    "EMIM.L",  "IQQQ.DE", "XDWD.DE", "DBXD.DE", "EEM",
    "SPY",     "QQQ",     "IVV",    "VTI",    "VOO",
    "GLD",     "SLV",     "TLT",    "HYG",    "LQD",
}

# Tickers delisted ou adquiridos — removidos do universo ML.
# Atualizar sempre que o S&P 500 / STOXX 600 tiver mudanças de composição.
# Fonte: yfinance "possibly delisted" errors nos logs do Railway.
_KNOWN_DELISTED: set[str] = {
    # Adquiridos (M&A) — já não existem como tickers independentes
    "DFS",     # Discover Financial → adquirido pela Capital One (2024)
    "HES",     # Hess → adquirido pela Chevron (2024)
    "JNPR",    # Juniper Networks → adquirido pela HPE (2024)
    "K",       # Kellanova → adquirido pela Mars (2024)
    "MRO",     # Marathon Oil → adquirido pela ConocoPhillips (2024)
    "PARA",    # Paramount → fusão/aquisição (2024)
    "WBA",     # Walgreens → going private (2024)
    "IPG",     # Interpublic → adquirido pela Omnicom (2024)
    "MMC",     # Marsh McLennan → verificar; pode ser ticker inalterado
    "SGEN",    # Seagen → adquirido pela Pfizer (2023)
    "BCR",     # Bard → adquirido pela BD (ticker BCR não mais existe)
    "WYDAY",   # Workday → ticker obsoleto
    # Europa — delisted
    "CSGN.SW", # Credit Suisse → adquirido pelo UBS (2023)
    "STM.MI",  # ST Micro Milan → verificar duplicado (STM é o ticker US)
    "FCA.MI",  # Fiat Chrysler → fusão em Stellantis (STLA)
    "CRH.I",   # CRH Dublin → migrado para NYSE (CRH)
    "KYGA.I",  # Kerry Group Ireland → ticker mudou
    # London
    "EVR.L",   # Evraz → delisted por sanções Rússia (2022)
    "SMDS.L",  # Smith DS → adquirido / ticker mudou
    "NEXT.L",  # Next Plc → verificar; pode ser ticker incorreto
    "PHNX.L",  # Phoenix Group → verificar ticker
    "WRK",     # WestRock → fusão com Smurfit (2024)
    "CTLT",    # Catalent → adquirido pela Novo Holdings (2024)
    "DAY",     # Dayforce (antigo Ceridian) → ticker mudou de CDAY
    "FI",      # Fiserv → ticker pode ter mudado para FI mas sem timezone = possível erro
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
    "EUNL.DE",
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
# Cache em disco — persiste entre reinícios do container Railway
# ─────────────────────────────────────────────────────────────────────────

def _save_universe_cache(tickers: list[str]) -> None:
    """Grava lista de tickers em /data/ml_universe_cache.json."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"date": date.today().isoformat(), "tickers": tickers}
        _CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        logging.info(f"[universe] Cache em disco actualizado: {len(tickers)} tickers → {_CACHE_FILE}")
    except Exception as e:
        logging.warning(f"[universe] Falha ao gravar cache em disco: {e}")


def _load_universe_cache() -> list[str]:
    """
    Carrega cache em disco se existir e tiver menos de _CACHE_DAYS dias.
    Devolve [] se não existir ou estiver expirado.
    FIX: se o cache existir mas estiver expirado, ainda o usa como fallback
    de emergência (evita snap vazio quando Wikipedia falha).
    """
    if not _CACHE_FILE.exists():
        return []
    try:
        payload = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        cached_date = date.fromisoformat(payload["date"])
        age = (date.today() - cached_date).days
        tickers = payload.get("tickers", [])
        if len(tickers) > 100:
            if age <= _CACHE_DAYS:
                logging.info(f"[universe] Cache em disco válido ({age}d): {len(tickers)} tickers")
            else:
                # FIX: cache expirado mas usável — melhor que vazio
                logging.warning(
                    f"[universe] Cache em disco expirado ({age}d) mas com {len(tickers)} tickers — "
                    f"a usar como fallback de emergência até Wikipedia estar disponível."
                )
            return tickers
        logging.info(f"[universe] Cache em disco pequeno ({len(tickers)} tickers), a reconstruir.")
        return []
    except Exception as e:
        logging.warning(f"[universe] Falha ao ler cache em disco: {e}")
        return []


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


# Cache em memória — invalidado se o resultado for vazio
_universe_cache: list[str] = []


def get_full_universe() -> list[str]:
    """
    Universo completo incluindo ETFs.
    Para scan de watchlist/preço. NÃO usar directamente no pipeline ML.

    Ordem de prioridade:
      1. Cache em memória (mesmo processo, não vazio)
      2. Cache em disco (/data/ml_universe_cache.json, <= 7 dias)
         FIX: cache expirado ainda usado como fallback de emergência
      3. Scraping Wikipedia + fallbacks estáticos
    Grava em disco sempre que constrói a lista de raiz.
    """
    global _universe_cache

    # 1. Cache em memória
    if _universe_cache:
        return _universe_cache

    # 2. Cache em disco (sobrevive a reinícios do container)
    # FIX: _load_universe_cache() já usa cache expirado como fallback
    disk = _load_universe_cache()
    if disk:
        _universe_cache = disk
        # Se o cache estava expirado, tentar refrescar em background seria ideal;
        # por ora, continuamos com o valor antigo e tentamos atualizar agora.
        # O snapshot do dia corre na mesma com tickers válidos.
        return _universe_cache

    # 3. Construir de raiz
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

    if len(unique) > 100:
        _universe_cache = unique
        _save_universe_cache(unique)
    else:
        logging.error(
            f"[universe] Resultado suspeito: apenas {len(unique)} tickers — "
            f"a usar fallbacks estáticos completos."
        )
        # Último recurso: usar só os fallbacks estáticos (sempre disponíveis)
        _universe_cache = list(dict.fromkeys(
            USER_PORTFOLIO + USER_WATCHLIST + _SP500_FALLBACK + _NASDAQ100_FALLBACK + _STOXX200 + _FTSE100
        ))
        # Gravar mesmo o fallback estático para que próxima chamada use o disco
        _save_universe_cache(_universe_cache)
        logging.info(f"[universe] Fallback estático: {len(_universe_cache)} tickers gravados em cache")

    return _universe_cache


def get_ml_universe() -> list[str]:
    """
    Universo filtrado para o pipeline ML.
    Remove ETFs — sem fundamentais válidos no yfinance.
    Usar em backtest.py, train_model.py, scan diurno.

    FIX: guard explícito contra lista vazia — levanta RuntimeError
    para que o chamador (universe_snapshot) saiba que algo falhou,
    em vez de silenciosamente processar 0 tickers.
    """
    full     = get_full_universe()
    filtered = [t for t in full if t not in ETF_TICKERS and t not in _KNOWN_DELISTED]
    excluded = len(full) - len(filtered)
    logging.info(f"[universe] ML universe: {len(filtered)} tickers ({excluded} ETFs excluídos)")

    # FIX: nunca retornar silenciosamente uma lista vazia
    if len(filtered) < 50:
        raise RuntimeError(
            f"[universe] get_ml_universe() devolveu apenas {len(filtered)} tickers — "
            f"scraping falhou e cache em disco inexistente. Verifica /data/ml_universe_cache.json."
        )

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
    "WYDAY","XEL","ZM","ZS",
]
