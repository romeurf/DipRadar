"""
insider_radar.py — DipRadar Insider & Signal Intelligence Module

Fontes:
  1. SEC EDGAR Form 4 (insider trades oficiais — gratuito, sem API key)
  2. OpenInsider.com scraping (agrega Form 4 com contexto extra)
  3. RSS/Google News scraping (sinais sociais antes de serem oficiais)
     — Trump tweets, declarações de membros do governo, etc.

Filosofia:
  • Cobertura TOTAL — nenhuma whitelist. Todos os insiders são reportados.
  • Identidade contextualizada — o bot diz-te QUEM é a pessoa e POR QUÊ importa.
  • Lógica de relevância cargo↔sector:
      Sec. Defesa compra empresa de defesa   → 🔴 MÁXIMO ALERTA
      Sec. Saúde compra empresa pharma       → 🔴 MÁXIMO ALERTA
      CEO compra ações da própria empresa    → 🟡 COMPRA CONVICTA
      Político genérico compra tech          → 🟢 NOTABLE
  • Filtro de montante: só reporta trades com valor ≥ $50k
    (configurável via env INSIDER_MIN_VALUE_USD).

Uso no main.py:
  from insider_radar import InsiderRadar
  radar = InsiderRadar(send_fn=send_telegram)
  radar.start()   # arranca thread de polling (cada 30min)
  radar.check_now()  # força scan imediato
"""

from __future__ import annotations

import os
import re
import time
import logging
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

_MIN_VALUE_USD: int = int(os.environ.get("INSIDER_MIN_VALUE_USD", 50_000))
_POLL_INTERVAL: int = int(os.environ.get("INSIDER_POLL_MINUTES", 30)) * 60
_SOCIAL_POLL:   int = int(os.environ.get("INSIDER_SOCIAL_MINUTES", 15)) * 60
_HEADERS = {
    "User-Agent": "DipRadar/2.0 research-bot contact@dipadar.dev",
    "Accept-Language": "en-US,en;q=0.9",
}

# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY DATABASE — Quem é quem
# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento nome → perfil. Pesquisa case-insensitive por substring.
# Adiciona entradas aqui à medida que novos insiders relevantes aparecem.

_IDENTITY_DB: list[dict] = [
    # ── Governo EUA (current & recent) ────────────────────────────────────────
    {"names": ["donald trump", "trump"], "role": "Presidente dos EUA",
     "sector_bias": [], "trust_tier": "GOVERNMENT",
     "note": "POTUS. Qualquer declaração pública sobre acções é mercado-mover."},
    {"names": ["pete hegseth"], "role": "Secretário da Defesa (EUA)",
     "sector_bias": ["Defense", "Aerospace"], "trust_tier": "CABINET",
     "note": "Acesso a contratos de defesa classificados. Compras no sector = 🔴"},
    {"names": ["robert kennedy", "rfk", "kennedy jr"], "role": "Secretário da Saúde (EUA / HHS)",
     "sector_bias": ["Healthcare", "Pharmaceuticals", "Biotechnology"], "trust_tier": "CABINET",
     "note": "Regula FDA, CMS. Compras pharma/biotech = 🔴 conflito máximo."},
    {"names": ["scott bessent"], "role": "Secretário do Tesouro (EUA)",
     "sector_bias": ["Financials", "Banks"], "trust_tier": "CABINET",
     "note": "Controla política fiscal e regulação bancária."},
    {"names": ["howard lutnick"], "role": "Secretário do Comércio (EUA)",
     "sector_bias": ["Technology", "Trade", "Semiconductors"], "trust_tier": "CABINET",
     "note": "Ex-CEO Cantor Fitzgerald. Política comercial e tarifas."},
    {"names": ["doug burgum"], "role": "Secretário do Interior (EUA)",
     "sector_bias": ["Energy", "Oil & Gas", "Mining"], "trust_tier": "CABINET",
     "note": "Supervisiona recursos naturais e concessões energéticas."},
    {"names": ["chris wright"], "role": "Secretário da Energia (EUA)",
     "sector_bias": ["Energy", "Oil & Gas", "Utilities"], "trust_tier": "CABINET",
     "note": "Ex-CEO Liberty Energy. Compras energia = 🔴"},
    {"names": ["marco rubio"], "role": "Secretário de Estado (EUA)",
     "sector_bias": ["Defense", "Technology"], "trust_tier": "CABINET",
     "note": "Política externa e exportações tecnológicas."},
    {"names": ["elon musk", "musk"], "role": "Diretor DOGE / Tesla CEO / SpaceX CEO",
     "sector_bias": ["Technology", "Automotive", "Aerospace", "Defense"], "trust_tier": "VIP",
     "note": "Market mover extremo. Posts no X movem mercados em minutos."},
    {"names": ["nancy pelosi", "pelosi"], "role": "Ex-Speaker Congresso (D-CA)",
     "sector_bias": [], "trust_tier": "CONGRESS",
     "note": "Historial de trades controversos. Alta cobertura mediática."},
    {"names": ["paul pelosi"], "role": "Empresário — cônjuge de Nancy Pelosi",
     "sector_bias": [], "trust_tier": "CONGRESS_FAMILY",
     "note": "Trades espelhados da cônjuge congressista."},
    {"names": ["michael burgess"], "role": "Congressista (R-TX) — médico",
     "sector_bias": ["Healthcare"], "trust_tier": "CONGRESS", "note": ""},
    # ── Tech Giants CEOs ───────────────────────────────────────────────────────
    {"names": ["jensen huang", "huang"], "role": "CEO NVIDIA",
     "sector_bias": ["Semiconductors", "Technology"], "trust_tier": "CEO",
     "note": "CEO e cofundador. Vendas são RSU programadas — compras são raras e muito significativas."},
    {"names": ["satya nadella"], "role": "CEO Microsoft",
     "sector_bias": ["Technology"], "trust_tier": "CEO", "note": ""},
    {"names": ["tim cook"], "role": "CEO Apple",
     "sector_bias": ["Technology"], "trust_tier": "CEO", "note": ""},
    {"names": ["sundar pichai"], "role": "CEO Alphabet/Google",
     "sector_bias": ["Technology"], "trust_tier": "CEO", "note": ""},
    {"names": ["mark zuckerberg", "zuckerberg"], "role": "CEO Meta",
     "sector_bias": ["Technology"], "trust_tier": "CEO", "note": ""},
    {"names": ["jeff bezos", "bezos"], "role": "Fundador Amazon",
     "sector_bias": ["Technology", "E-Commerce"], "trust_tier": "CEO", "note": ""},
    {"names": ["sam altman"], "role": "CEO OpenAI",
     "sector_bias": ["Technology", "AI"], "trust_tier": "CEO",
     "note": "CEO da empresa mais influente em IA. Raramente tem posições públicas."},
    # ── Investidores institucionais conhecidos ────────────────────────────────
    {"names": ["warren buffett", "buffett"], "role": "CEO Berkshire Hathaway",
     "sector_bias": [], "trust_tier": "LEGEND",
     "note": "Oracle of Omaha. 13F/Form 4 movem mercados."},
    {"names": ["charlie munger", "munger"], "role": "Ex-Vice-Chairman Berkshire",
     "sector_bias": [], "trust_tier": "LEGEND", "note": ""},
    {"names": ["bill ackman", "ackman"], "role": "CEO Pershing Square",
     "sector_bias": [], "trust_tier": "ACTIVIST",
     "note": "Ativista vocal. Anuncia posições em X para criar pressão."},
    {"names": ["carl icahn", "icahn"], "role": "Investidor Ativista",
     "sector_bias": [], "trust_tier": "ACTIVIST", "note": ""},
    {"names": ["michael burry", "burry"], "role": "CEO Scion Asset Management",
     "sector_bias": [], "trust_tier": "LEGEND",
     "note": "Big Short. 13F trimestrais são muito seguidas."},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR KEYWORD MAP — para detectar sobreposição cargo↔empresa
# ─────────────────────────────────────────────────────────────────────────────

_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Defense": ["defense", "defence", "lockheed", "raytheon", "northrop",
                "general dynamics", "l3harris", "booz allen", "leidos",
                "mantech", "military", "weapons", "aircraft", "missile"],
    "Aerospace": ["aerospace", "boeing", "airbus", "spacex", "rocket",
                  "satellite", "aviation"],
    "Healthcare": ["health", "hospital", "hca", "unitedhealth", "cigna",
                   "cvs", "walgreen", "mckesson", "cardinal"],
    "Pharmaceuticals": ["pharma", "pfizer", "eli lilly", "abbvie", "bristol",
                        "merck", "novartis", "roche", "astrazeneca", "sanofi"],
    "Biotechnology": ["biotech", "biogen", "moderna", "biontech", "regeneron",
                      "gilead", "amgen", "vertex", "crispr", "gene"],
    "Financials": ["bank", "financial", "jpmorgan", "goldman", "morgan stanley",
                   "wells fargo", "citigroup", "blackrock", "vanguard",
                   "insurance", "asset management"],
    "Technology": ["tech", "software", "cloud", "microsoft", "apple", "google",
                   "amazon", "meta", "nvidia", "intel", "oracle", "salesforce",
                   "servicenow", "palantir", "databricks", "ai", "cyber"],
    "Semiconductors": ["semiconductor", "chip", "nvidia", "amd", "intel",
                       "qualcomm", "broadcom", "tsmc", "asml", "marvell",
                       "micron", "applied materials"],
    "Energy": ["energy", "oil", "gas", "exxon", "chevron", "shell", "bp",
               "pioneer", "conocophillips", "halliburton", "schlumberger",
               "solar", "wind", "renewabl"],
    "Oil & Gas": ["oil", "gas", "drilling", "pipeline", "refin"],
    "Utilities": ["utility", "utilities", "electric", "power", "grid",
                  "nextera", "duke", "dominion"],
    "Mining": ["mining", "gold", "silver", "copper", "lithium", "uranium",
               "barrick", "newmont", "freeport"],
    "Automotive": ["auto", "car", "vehicle", "tesla", "ford", "gm", "toyota",
                   "rivian", "ev", "electric vehicle"],
    "E-Commerce": ["e-commerce", "ecommerce", "retail", "amazon", "shopify",
                   "ebay", "etsy"],
    "Trade": ["trade", "tariff", "import", "export", "supply chain"],
    "Banks": ["bank", "banking", "jpmorgan", "goldman", "wells", "citi",
              "bofa", "bank of america"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL SIGNAL SOURCES — RSS feeds que capturam sinais antes de serem oficiais
# ─────────────────────────────────────────────────────────────────────────────

_SOCIAL_SOURCES: list[dict] = [
    {
        "name": "Trump Truth Social via Truthsocial.com RSS",
        "url": "https://truthsocial.com/@realDonaldTrump/feed.rss",
        "person": "trump",
        "type": "social",
    },
    {
        "name": "MarketWatch Breaking News",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "person": None,
        "type": "news",
    },
    {
        "name": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "person": None,
        "type": "news",
    },
    {
        "name": "CNBC Top News",
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "person": None,
        "type": "news",
    },
    {
        "name": "Bloomberg Politics RSS",
        "url": "https://feeds.bloomberg.com/politics/news.rss",
        "person": None,
        "type": "news",
    },
]

# Keywords que, em combinação com tickers/empresas, disparam alerta de sinal social
_SOCIAL_BUY_KEYWORDS = [
    "bought", "buying", "purchased", "invested", "great company",
    "fantastic stock", "invest in", "we bought", "i own", "long",
    "comprei", "fantastic", "incredible", "tremendous", "winner",
    "announce", "contract", "awarded", "billion dollar deal",
    "executive order", "sanctions", "ban", "restrict",
]

_SOCIAL_SELL_KEYWORDS = [
    "sold", "selling", "divesting", "short", "terrible company",
    "avoid", "bankrupt", "fraud", "investigation", "probe", "ban",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_identity(name: str) -> Optional[dict]:
    """Procura identidade na base de dados por substring case-insensitive."""
    name_lower = name.lower().strip()
    for entry in _IDENTITY_DB:
        for alias in entry["names"]:
            if alias in name_lower or name_lower in alias:
                return entry
    return None


def _sector_overlap(identity: dict, company_name: str, ticker: str) -> bool:
    """Verifica se o sector do cargo do insider se sobrepõe com a empresa comprada."""
    if not identity:
        return False
    combined = (company_name + " " + ticker).lower()
    for sector in identity.get("sector_bias", []):
        keywords = _SECTOR_KEYWORDS.get(sector, [])
        for kw in keywords:
            if kw in combined:
                return True
    return False


def _alert_level(identity: Optional[dict], value_usd: float,
                 is_sector_overlap: bool, transaction_type: str) -> str:
    """Calcula nível de alerta: EXTREME / HIGH / MEDIUM / LOW"""
    if not identity:
        if value_usd >= 5_000_000:
            return "HIGH"
        if value_usd >= 1_000_000:
            return "MEDIUM"
        return "LOW"

    tier = identity.get("trust_tier", "UNKNOWN")
    if tier == "GOVERNMENT" and is_sector_overlap:
        return "EXTREME"
    if tier in ("CABINET",) and is_sector_overlap:
        return "EXTREME"
    if tier in ("CABINET", "GOVERNMENT"):
        return "HIGH"
    if tier in ("LEGEND", "ACTIVIST", "VIP"):
        return "HIGH" if value_usd >= 1_000_000 else "MEDIUM"
    if tier in ("CEO",) and transaction_type == "P":
        return "HIGH" if value_usd >= 500_000 else "MEDIUM"
    if tier in ("CONGRESS", "CONGRESS_FAMILY"):
        return "HIGH" if is_sector_overlap else "MEDIUM"
    return "LOW"


def _format_value(v: float) -> str:
    """Formata valor em USD legível."""
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


def _emoji_level(level: str, t_type: str) -> str:
    action_em = "🟢" if t_type == "P" else "🔴"
    if level == "EXTREME":
        return f"🚨🚨🚨 {action_em}"
    if level == "HIGH":
        return f"⚠️ {action_em}"
    if level == "MEDIUM":
        return f"👀 {action_em}"
    return f"📋 {action_em}"


def _hash_item(text: str) -> str:
    """Gera hash curta para deduplicação."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR FORM 4 PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_edgar_form4(lookback_hours: int = 48) -> list[dict]:
    """
    Scrapa os últimos Form 4 do SEC EDGAR EFTS (full-text search).
    Retorna lista de trades desnormalizados.
    """
    try:
        url = (
            "https://efts.sec.gov/LATEST/search-index?q=%224%22"
            "&dateRange=custom"
            f"&startdt={(datetime.utcnow() - timedelta(hours=lookback_hours)).strftime('%Y-%m-%d')}"
            "&forms=4"
            "&hits.hits.total.value=true"
            "&hits.hits._source.period_of_report=true"
        )
        # Usa o endpoint RSS do EDGAR que é mais simples e estável
        rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&dateb=&owner=include&count=40&search_text=&output=atom"
        r = requests.get(rss_url, headers=_HEADERS, timeout=15)
        if not r.ok:
            log.warning(f"[edgar] HTTP {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, "xml")
        entries = soup.find_all("entry")
        trades: list[dict] = []

        for entry in entries[:40]:
            try:
                title_tag = entry.find("title")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                # Formato típico: "4 - Companyname (TICKER) (Issuer) — Ownername"
                m_ticker = re.search(r"\(([A-Z]{1,5})\)", title)
                ticker = m_ticker.group(1) if m_ticker else ""

                link_tag = entry.find("link")
                link = link_tag.get("href", "") if link_tag else ""

                updated_tag = entry.find("updated")
                date_str = updated_tag.get_text(strip=True)[:10] if updated_tag else ""

                # Tenta parsear o filing XML para obter detalhes reais
                trade_details = _parse_form4_filing(link, ticker)
                for td in trade_details:
                    td["filing_date"] = date_str
                    trades.append(td)

            except Exception as e:
                log.debug(f"[edgar] entry parse error: {e}")
                continue

        return trades

    except Exception as e:
        log.error(f"[edgar] fetch error: {e}")
        return []


def _parse_form4_filing(filing_url: str, ticker: str) -> list[dict]:
    """
    Faz parse do XML do Form 4 para extrair:
    - Nome do insider
    - Título/cargo
    - Tipo de transacção (P=compra, S=venda)
    - Quantidade + preço + valor total
    """
    trades: list[dict] = []
    try:
        # O link do RSS aponta para o HTML do filing; precisamos do XML
        # Converte URL do filing HTML → índice do filing
        # https://www.sec.gov/Archives/edgar/data/.../000.../form4.xml
        r = requests.get(filing_url, headers=_HEADERS, timeout=10)
        if not r.ok:
            return trades

        soup = BeautifulSoup(r.text, "html.parser")
        # Procura link para o ficheiro XML principal
        xml_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".xml" in href.lower() and "edgar" in href.lower():
                xml_link = "https://www.sec.gov" + href if href.startswith("/") else href
                break

        if not xml_link:
            return trades

        rx = requests.get(xml_link, headers=_HEADERS, timeout=10)
        if not rx.ok:
            return trades

        xsoup = BeautifulSoup(rx.text, "xml")

        # Nome e cargo do insider
        name_tag = xsoup.find("rptOwnerName")
        insider_name = name_tag.get_text(strip=True) if name_tag else "Unknown"

        title_tag = xsoup.find("officerTitle")
        insider_title = title_tag.get_text(strip=True) if title_tag else ""

        issuer_tag = xsoup.find("issuerName")
        company_name = issuer_tag.get_text(strip=True) if issuer_tag else ""

        ticker_tag = xsoup.find("issuerTradingSymbol")
        if ticker_tag:
            ticker = ticker_tag.get_text(strip=True)

        # Transacções não-derivativas (acções directas)
        for txn in xsoup.find_all("nonDerivativeTransaction"):
            try:
                t_code_tag = txn.find("transactionCode")
                t_code = t_code_tag.get_text(strip=True) if t_code_tag else ""
                if t_code not in ("P", "S"):  # P=compra, S=venda
                    continue

                shares_tag = txn.find("transactionShares")
                price_tag  = txn.find("transactionPricePerShare")
                shares = float(shares_tag.find("value").get_text()) if shares_tag and shares_tag.find("value") else 0.0
                price  = float(price_tag.find("value").get_text())  if price_tag  and price_tag.find("value")  else 0.0
                value  = shares * price

                if value < _MIN_VALUE_USD:
                    continue

                trades.append({
                    "insider_name":  insider_name,
                    "insider_title": insider_title,
                    "ticker":        ticker,
                    "company_name":  company_name,
                    "transaction":   t_code,
                    "shares":        shares,
                    "price":         price,
                    "value_usd":     value,
                    "source":        "SEC Form 4",
                    "url":           xml_link,
                })
            except Exception as e:
                log.debug(f"[form4 txn] {e}")
                continue

    except Exception as e:
        log.debug(f"[form4 parse] {e}")

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# OPENINSIDER SCRAPER (backup / validação cruzada)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_openinsider(lookback_days: int = 2) -> list[dict]:
    """
    Scrapa OpenInsider.com — agrega Form 4 com detalhes extras como
    percentagem da participação comprada e título do insider.
    Endpoint: https://openinsider.com/screener com filtros GET.
    """
    try:
        url = (
            "https://openinsider.com/screener?"
            "s=&o=&pl=&ph=&ll=&lh=&fd=-2&fdr=&td=0&tdr=&fdlyl=&fdlyh=&"
            f"daysago={lookback_days}&xp=1&xs=1"
            "&vl=50&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999"
            "&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
            "&sortcol=0&cnt=100&page=1"
        )
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if not r.ok:
            log.warning(f"[openinsider] HTTP {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            return []

        trades: list[dict] = []
        rows = table.find_all("tr")[1:]  # pula header
        for row in rows:
            try:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < 13:
                    continue

                # Colunas OpenInsider (ordem típica):
                # 0=X, 1=Filing Date, 2=Trade Date, 3=Ticker, 4=Company,
                # 5=Insider Name, 6=Title, 7=Trade Type, 8=Price,
                # 9=Qty, 10=Owned, 11=dOwned%, 12=Value
                date_str    = cells[1]
                ticker      = cells[3].upper()
                company     = cells[4]
                name        = cells[5]
                title       = cells[6]
                t_type_raw  = cells[7]  # "P - Purchase", "S - Sale"
                price_str   = cells[8].replace(",", "").replace("$", "")
                qty_str     = cells[9].replace(",", "").replace("+", "")
                value_str   = cells[12].replace(",", "").replace("$", "").replace("+", "")

                t_code = "P" if "P" in t_type_raw.upper() else ("S" if "S" in t_type_raw.upper() else "")
                if not t_code:
                    continue

                try:
                    value = float(value_str) if value_str else 0.0
                except ValueError:
                    value = 0.0

                if value < _MIN_VALUE_USD:
                    continue

                try:
                    price = float(price_str) if price_str else 0.0
                except ValueError:
                    price = 0.0

                try:
                    qty = float(qty_str) if qty_str else 0.0
                except ValueError:
                    qty = 0.0

                trades.append({
                    "insider_name":  name,
                    "insider_title": title,
                    "ticker":        ticker,
                    "company_name":  company,
                    "transaction":   t_code,
                    "shares":        qty,
                    "price":         price,
                    "value_usd":     value,
                    "filing_date":   date_str,
                    "source":        "OpenInsider",
                    "url":           f"https://openinsider.com/{ticker.lower()}",
                })
            except Exception as e:
                log.debug(f"[openinsider row] {e}")
                continue

        return trades

    except Exception as e:
        log.error(f"[openinsider] {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL / NEWS SIGNAL SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_social_signals() -> list[dict]:
    """
    Monitoriza RSS feeds de notícias e redes sociais à procura de sinais
    de compra/venda ANTES de serem registados no SEC.
    """
    signals: list[dict] = []

    for source in _SOCIAL_SOURCES:
        try:
            r = requests.get(source["url"], headers=_HEADERS, timeout=10)
            if not r.ok:
                continue

            soup = BeautifulSoup(r.text, "xml")
            # Suporta RSS e Atom
            items = soup.find_all("item") or soup.find_all("entry")

            for item in items[:30]:
                title_tag = item.find("title")
                desc_tag  = item.find("description") or item.find("summary") or item.find("content")
                link_tag  = item.find("link")
                pub_tag   = item.find("pubDate") or item.find("published") or item.find("updated")

                title    = title_tag.get_text(strip=True) if title_tag else ""
                desc     = desc_tag.get_text(strip=True)[:500] if desc_tag else ""
                link     = link_tag.get_text(strip=True) if link_tag else ""
                pub_date = pub_tag.get_text(strip=True)[:16] if pub_tag else ""
                combined = (title + " " + desc).lower()

                # Detecta sinal de compra/venda
                buy_match  = any(kw in combined for kw in _SOCIAL_BUY_KEYWORDS)
                sell_match = any(kw in combined for kw in _SOCIAL_SELL_KEYWORDS)
                if not buy_match and not sell_match:
                    continue

                # Detecta menção a tickers (maiúsculas 1-5 letras precedidas de $ ou isoladas)
                tickers_mentioned = re.findall(
                    r'\$([A-Z]{1,5})|\b([A-Z]{2,5})\b(?=\s+stock|\s+shares|\s+equity)',
                    title + " " + desc
                )
                tickers_mentioned = [
                    t[0] or t[1] for t in tickers_mentioned if any(t)
                ]

                # Detecta nomes conhecidos na notícia
                mentioned_person = None
                if source["person"]:
                    # É um feed pessoal — a pessoa é sempre a do feed
                    for entry in _IDENTITY_DB:
                        if source["person"] in entry["names"]:
                            mentioned_person = entry
                            break
                else:
                    # Feed de notícias — procura nomes na headline
                    for entry in _IDENTITY_DB:
                        for alias in entry["names"]:
                            if alias in combined and len(alias) > 4:  # evita falsos positivos curtos
                                mentioned_person = entry
                                break
                        if mentioned_person:
                            break

                signal_type = "BUY_SIGNAL" if buy_match else "SELL_SIGNAL"

                signals.append({
                    "type":       signal_type,
                    "title":      title[:200],
                    "summary":    desc[:300],
                    "tickers":    tickers_mentioned[:5],
                    "person":     mentioned_person,
                    "source":     source["name"],
                    "url":        link,
                    "date":       pub_date,
                })

        except Exception as e:
            log.debug(f"[social] {source['name']}: {e}")
            continue

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _format_trade_message(trade: dict) -> str:
    """
    Formata mensagem Telegram para um trade de insider.
    Inclui identidade, contexto de cargo e nível de alerta.
    """
    name     = trade.get("insider_name", "Unknown")
    title    = trade.get("insider_title", "")
    ticker   = trade.get("ticker", "")
    company  = trade.get("company_name", "")
    t_type   = trade.get("transaction", "")
    value    = trade.get("value_usd", 0)
    price    = trade.get("price", 0)
    shares   = trade.get("shares", 0)
    date_str = trade.get("filing_date", "")
    source   = trade.get("source", "")
    url      = trade.get("url", "")

    identity  = _get_identity(name)
    overlap   = _sector_overlap(identity, company, ticker)
    level     = _alert_level(identity, value, overlap, t_type)
    emoji_hdr = _emoji_level(level, t_type)
    action    = "COMPRA" if t_type == "P" else "VENDA"

    lines = [f"{emoji_hdr} *INSIDER {action} — {ticker}*"]

    # Quem é esta pessoa?
    if identity:
        role = identity.get("role", "")
        note = identity.get("note", "")
        tier = identity.get("trust_tier", "")
        tier_badge = {
            "GOVERNMENT": "🏛️ GOVERNO",
            "CABINET":    "🏛️ MINISTRO",
            "CONGRESS":   "🏛️ CONGRESSO",
            "CONGRESS_FAMILY": "🏛️ FAMÍLIA CONGRESSO",
            "LEGEND":     "🧙 LENDA",
            "ACTIVIST":   "⚔️ ATIVISTA",
            "VIP":        "⭐ VIP",
            "CEO":        "👔 CEO",
        }.get(tier, "👤")
        lines.append(f"  *{name}*  {tier_badge}")
        lines.append(f"  _{role}_")
        if note:
            lines.append(f"  💬 _{note}_")
    else:
        lines.append(f"  *{name}*  👤 _{title or 'Insider'}_")
        lines.append(f"  _Insider sem perfil conhecido na base de dados._")
        if title:
            lines.append(f"  Cargo declarado: _{title}_")

    lines.append("")
    lines.append(f"  🏢 *{company}* (`{ticker}`")
    if price > 0:
        lines.append(f"  💰 *{_format_value(value)}* | {shares:,.0f} acções @ ${price:.2f}")
    else:
        lines.append(f"  💰 *{_format_value(value)}*")

    if date_str:
        lines.append(f"  📅 {date_str}")

    # Contexto de relevância
    if overlap and identity:
        biases = identity.get("sector_bias", [])
        lines.append("")
        lines.append(f"  🔗 *SOBREPOSIÇÃO CARGO↔SECTOR* ({', '.join(biases)})")
        lines.append("  _Este insider tem autoridade directa sobre o sector desta empresa!_")

    lines.append(f"  📌 Fonte: {source}")
    if url:
        lines.append(f"  🔗 [Ver filing]({url})")

    # Nível de alerta como rodapé
    level_note = {
        "EXTREME": "🚨 ALERTA MÁXIMO — cargo com influência directa no sector",
        "HIGH":    "⚠️ ALERTA ALTO — pessoa de alto perfil",
        "MEDIUM":  "👀 NOTABLE — vale a pena seguir",
        "LOW":     "📋 Registo rotineiro",
    }.get(level, "")
    if level_note:
        lines.append(f"  {level_note}")

    return "\n".join(lines)


def _format_social_message(signal: dict) -> str:
    """
    Formata mensagem para sinal social/notícia.
    """
    s_type   = signal.get("type", "")
    title    = signal.get("title", "")
    summary  = signal.get("summary", "")
    tickers  = signal.get("tickers", [])
    person   = signal.get("person")
    source   = signal.get("source", "")
    url      = signal.get("url", "")
    date     = signal.get("date", "")

    action_em = "📈" if s_type == "BUY_SIGNAL" else "📉"
    action    = "SINAL BULLISH" if s_type == "BUY_SIGNAL" else "SINAL BEARISH"

    lines = [f"{action_em} *{action} — SINAL SOCIAL/NOTÍCIA*"]

    if person:
        role = person.get("role", "")
        name = person["names"][0].title()
        note = person.get("note", "")
        tier = person.get("trust_tier", "")
        tier_badge = {
            "GOVERNMENT": "🏛️ GOVERNO",
            "CABINET":    "🏛️ MINISTRO",
            "CONGRESS":   "🏛️ CONGRESSO",
            "LEGEND":     "🧙 LENDA",
            "ACTIVIST":   "⚔️ ATIVISTA",
            "VIP":        "⭐ VIP",
            "CEO":        "👔 CEO",
        }.get(tier, "👤")
        lines.append(f"  *{name}*  {tier_badge}")
        lines.append(f"  _{role}_")
        if note:
            lines.append(f"  💬 _{note}_")
    lines.append("")

    lines.append(f"  📰 *{title}*")
    if summary and summary != title:
        lines.append(f"  _{summary[:200]}_")

    if tickers:
        lines.append(f"  🎯 Tickers mencionados: *{', '.join(tickers)}*")

    if date:
        lines.append(f"  📅 {date}")
    lines.append(f"  📌 Fonte: {source}")
    if url:
        lines.append(f"  🔗 [Ver notícia]({url})")

    lines.append("")
    lines.append("  ⚡ _Sinal captado ANTES de filing oficial._")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RADAR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class InsiderRadar:
    """
    Radar de insiders para o DipRadar.

    Exemplo:
        radar = InsiderRadar(send_fn=send_telegram)
        radar.start()   # polling automático
        radar.check_now()  # scan imediato
    """

    def __init__(self, send_fn):
        self._send        = send_fn
        self._seen_trades: set[str] = set()
        self._seen_social: set[str] = set()
        self._thread_form4:  Optional[threading.Thread] = None
        self._thread_social: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Arranca threads de polling em background."""
        if self._running:
            return
        self._running = True
        self._thread_form4  = threading.Thread(target=self._loop_form4,  daemon=True, name="insider-form4")
        self._thread_social = threading.Thread(target=self._loop_social, daemon=True, name="insider-social")
        self._thread_form4.start()
        self._thread_social.start()
        log.info("[InsiderRadar] Iniciado (Form4 + Social).")

    def stop(self) -> None:
        self._running = False

    def check_now(self) -> None:
        """Força scan imediato (chama em thread separada)."""
        threading.Thread(target=self._scan_form4, daemon=True).start()
        threading.Thread(target=self._scan_social, daemon=True).start()

    # ── Loops de polling ──────────────────────────────────────────────────────

    def _loop_form4(self) -> None:
        # Espera inicial para deixar o bot arrancar
        time.sleep(60)
        while self._running:
            try:
                self._scan_form4()
            except Exception as e:
                log.error(f"[InsiderRadar form4 loop] {e}")
            time.sleep(_POLL_INTERVAL)

    def _loop_social(self) -> None:
        time.sleep(30)
        while self._running:
            try:
                self._scan_social()
            except Exception as e:
                log.error(f"[InsiderRadar social loop] {e}")
            time.sleep(_SOCIAL_POLL)

    # ── Scan methods ──────────────────────────────────────────────────────────

    def _scan_form4(self) -> None:
        """Scan Form 4 via SEC EDGAR + OpenInsider."""
        log.info("[InsiderRadar] Scanning Form 4...")

        # Prioridade: OpenInsider (mais completo e parseável)
        trades = _fetch_openinsider(lookback_days=2)
        if not trades:
            # Fallback para EDGAR directo
            trades = _fetch_edgar_form4(lookback_hours=48)

        sent = 0
        for trade in trades:
            try:
                uid = _hash_item(
                    f"{trade.get('insider_name','')}"
                    f"{trade.get('ticker','')}"
                    f"{trade.get('filing_date','')}"
                    f"{trade.get('transaction','')}"
                    f"{trade.get('value_usd', 0):.0f}"
                )
                if uid in self._seen_trades:
                    continue
                self._seen_trades.add(uid)

                msg = _format_trade_message(trade)
                self._send(msg)
                sent += 1
                time.sleep(0.5)  # Não spammar o Telegram

            except Exception as e:
                log.debug(f"[scan_form4 trade] {e}")

        if sent > 0:
            log.info(f"[InsiderRadar] {sent} novos trades enviados.")
        else:
            log.debug("[InsiderRadar] Sem novos trades.")

        # Limpa seen_trades antigo (mantém últimos 2000 para evitar RAM leak)
        if len(self._seen_trades) > 2000:
            self._seen_trades = set(list(self._seen_trades)[-1000:])

    def _scan_social(self) -> None:
        """Scan sinais sociais e notícias."""
        log.debug("[InsiderRadar] Scanning social signals...")

        signals = _fetch_social_signals()
        sent = 0

        for sig in signals:
            try:
                uid = _hash_item(
                    sig.get("title", "") + sig.get("date", "") + sig.get("source", "")
                )
                if uid in self._seen_social:
                    continue
                self._seen_social.add(uid)

                # Só envia sinais com pessoa conhecida OU tickers mencionados
                has_person  = sig.get("person") is not None
                has_tickers = bool(sig.get("tickers"))
                if not has_person and not has_tickers:
                    continue

                # Filtra por relevância: pessoa VIP/CABINET/GOVERNMENT tem prioridade
                person = sig.get("person")
                if person:
                    tier = person.get("trust_tier", "")
                    if tier not in ("GOVERNMENT", "CABINET", "CONGRESS", "LEGEND", "ACTIVIST", "VIP", "CEO"):
                        continue

                msg = _format_social_message(sig)
                self._send(msg)
                sent += 1
                time.sleep(0.5)

            except Exception as e:
                log.debug(f"[scan_social sig] {e}")

        if sent > 0:
            log.info(f"[InsiderRadar social] {sent} novos sinais enviados.")

        if len(self._seen_social) > 1000:
            self._seen_social = set(list(self._seen_social)[-500:])


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _print(msg):
        print("\n" + "="*60)
        print(msg)

    print("[TEST] InsiderRadar standalone test")
    print("[TEST] Fetching OpenInsider...")
    trades = _fetch_openinsider(lookback_days=3)
    print(f"[TEST] {len(trades)} trades encontrados")
    for t in trades[:5]:
        print(_format_trade_message(t))
        print("-" * 40)

    print("\n[TEST] Fetching social signals...")
    signals = _fetch_social_signals()
    print(f"[TEST] {len(signals)} signals encontrados")
    for s in signals[:3]:
        print(_format_social_message(s))
        print("-" * 40)
