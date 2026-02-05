"""
Stock Ticker List Module

Fetches stock tickers from multiple sources:
- FMP API (primary): S&P 500, Nasdaq 100, Dow Jones
- Wikipedia (fallback): S&P 500, MidCap 400, SmallCap 600
- ETF Holdings: Russell 2000 (via IWM)
- Curated lists: Additional small/mid caps
- Portfolio: Always included in scans
"""

import requests
from bs4 import BeautifulSoup
import logging
import os

logger = logging.getLogger(__name__)

# FMP API configuration
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Cache for ticker lists (avoid re-fetching on every scan)
_ticker_cache = {
    'sp500': None,
    'nasdaq100': None,
    'dowjones': None,
    'midcap400': None,
    'smallcap600': None,
    'russell2000': None,
    'last_fetch': None
}
CACHE_DURATION_HOURS = 24  # Refresh lists once per day


def get_all_tickers(include_portfolio: bool = True, exclude_delisted: bool = True) -> list[str]:
    """
    Get combined list of all major index tickers plus portfolio holdings.

    Includes:
    - S&P 500 (large cap) - from FMP or Wikipedia
    - S&P MidCap 400 (mid cap) - from Wikipedia
    - S&P SmallCap 600 (small cap) - from Wikipedia
    - Nasdaq 100 (tech-heavy) - from FMP
    - Russell 2000 (small cap) - from ETF holdings or curated
    - Dow Jones 30 (blue chips) - from FMP
    - Portfolio tickers (always scanned first)

    Args:
        include_portfolio: Include user's portfolio tickers
        exclude_delisted: Filter out tickers marked as delisted/invalid
    """
    # Get delisted tickers to exclude
    delisted = set()
    if exclude_delisted:
        try:
            from data_fetcher import get_delisted_tickers
            delisted = get_delisted_tickers()
            if delisted:
                logger.info(f"Excluding {len(delisted)} delisted/invalid tickers from scan")
        except Exception as e:
            logger.debug(f"Could not load delisted tickers: {e}")

    # Start with portfolio tickers (highest priority - never excluded)
    combined = []
    if include_portfolio:
        portfolio = get_portfolio_tickers()
        combined.extend(portfolio)

    # Fetch from all sources
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    dowjones = get_dowjones_tickers()
    midcap400 = get_sp400_midcap_tickers()
    smallcap600 = get_sp600_smallcap_tickers()
    russell2000 = get_russell2000_tickers()

    # Add index tickers
    combined.extend(sp500)
    combined.extend(nasdaq100)
    combined.extend(dowjones)
    combined.extend(midcap400)
    combined.extend(smallcap600)
    combined.extend(russell2000)

    # Remove duplicates while preserving order (portfolio first)
    # Also filter out delisted tickers (but keep portfolio tickers)
    seen = set()
    unique = []
    portfolio_set = set(get_portfolio_tickers()) if include_portfolio else set()

    for ticker in combined:
        if ticker and ticker not in seen:
            seen.add(ticker)
            # Keep portfolio tickers even if delisted, filter others
            if ticker in portfolio_set or ticker not in delisted:
                unique.append(ticker)

    return unique


def get_portfolio_tickers() -> list[str]:
    """
    Get tickers from the user's portfolio to ensure they're always scanned.
    Fetches from database if available, falls back to CSV.
    """
    tickers = []

    # Try to get from database first
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        from database import SessionLocal, PortfolioPosition

        db = SessionLocal()
        try:
            positions = db.query(PortfolioPosition.ticker).distinct().all()
            tickers = [p.ticker for p in positions]
            if tickers:
                logger.info(f"Loaded {len(tickers)} portfolio tickers from database")
                return tickers
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Could not load portfolio from database: {e}")

    # Fall back to CSV
    try:
        from pathlib import Path
        csv_path = Path(__file__).parent / "portfolio.csv"
        if csv_path.exists():
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                tickers = [row['ticker'] for row in reader if row.get('ticker')]
            logger.info(f"Loaded {len(tickers)} portfolio tickers from CSV")
    except Exception as e:
        logger.debug(f"Could not load portfolio from CSV: {e}")

    return tickers


def get_sp500_tickers() -> list[str]:
    """
    Fetch S&P 500 tickers from FMP API (primary) or Wikipedia (fallback).
    """
    # Try FMP API first (most reliable and up-to-date)
    if FMP_API_KEY:
        try:
            url = f"{FMP_BASE_URL}/sp500_constituent?apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                tickers = [item.get('symbol') for item in data if item.get('symbol')]
                if tickers:
                    logger.info(f"Fetched {len(tickers)} S&P 500 tickers from FMP")
                    return tickers
        except Exception as e:
            logger.warning(f"FMP S&P 500 fetch failed: {e}")

    # Fallback to Wikipedia
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})

        if table is None:
            table = soup.find('table', {'class': 'wikitable'})

        tickers = []
        rows = table.find_all('tr')[1:]  # Skip header

        for row in rows:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.replace('.', '-')  # BRK.B -> BRK-B
                tickers.append(ticker)

        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers

    except Exception as e:
        logger.warning(f"Wikipedia S&P 500 fetch failed: {e}")
        return get_fallback_tickers()


def get_nasdaq100_tickers() -> list[str]:
    """
    Fetch Nasdaq 100 tickers from FMP API (primary), Wikipedia (secondary), or hardcoded (fallback).
    """
    # Try FMP API first
    if FMP_API_KEY:
        try:
            url = f"{FMP_BASE_URL}/nasdaq_constituent?apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                tickers = [item.get('symbol') for item in data if item.get('symbol')]
                if tickers:
                    logger.info(f"Fetched {len(tickers)} Nasdaq 100 tickers from FMP")
                    return tickers
        except Exception as e:
            logger.warning(f"FMP Nasdaq 100 fetch failed: {e}")

    # Try Wikipedia scraping
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the constituents table (has 'Symbol' in header)
        tables = soup.find_all('table', {'class': 'wikitable'})

        for table in tables:
            header_row = table.find('tr')
            if header_row:
                headers = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
                if 'ticker' in headers or 'symbol' in headers:
                    # Found the right table
                    ticker_col = headers.index('ticker') if 'ticker' in headers else headers.index('symbol')
                    tickers = []
                    for row in table.find_all('tr')[1:]:  # Skip header
                        cells = row.find_all(['td', 'th'])
                        if len(cells) > ticker_col:
                            ticker = cells[ticker_col].text.strip()
                            ticker = ticker.replace('.', '-')  # BRK.B -> BRK-B
                            if ticker and ticker.isalpha():
                                tickers.append(ticker)

                    if len(tickers) >= 90:  # Nasdaq 100 should have ~100-103 tickers
                        logger.info(f"Fetched {len(tickers)} Nasdaq 100 tickers from Wikipedia")
                        return tickers

    except Exception as e:
        logger.warning(f"Wikipedia Nasdaq 100 fetch failed: {e}")

    # Fallback to hardcoded list
    logger.info("Using hardcoded Nasdaq 100 list")
    return get_fallback_nasdaq100_tickers()


def get_fallback_nasdaq100_tickers() -> list[str]:
    """Fallback Nasdaq 100 tickers."""
    return [
        "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
        "AMZN", "ANSS", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CCEP",
        "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP",
        "CSX", "CTAS", "CTSH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST",
        "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC",
        "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP",
        "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX",
        "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
        "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS", "TEAM", "TMUS",
        "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
    ]


def get_dowjones_tickers() -> list[str]:
    """
    Fetch Dow Jones 30 tickers from FMP API (primary), Wikipedia (secondary), or hardcoded (fallback).
    """
    # Try FMP API first
    if FMP_API_KEY:
        try:
            url = f"{FMP_BASE_URL}/dowjones_constituent?apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                tickers = [item.get('symbol') for item in data if item.get('symbol')]
                if tickers:
                    logger.info(f"Fetched {len(tickers)} Dow Jones tickers from FMP")
                    return tickers
        except Exception as e:
            logger.warning(f"FMP Dow Jones fetch failed: {e}")

    # Try Wikipedia scraping
    try:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the constituents table
        tables = soup.find_all('table', {'class': 'wikitable'})

        for table in tables:
            header_row = table.find('tr')
            if header_row:
                headers_text = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
                # Look for table with 'symbol' or 'ticker' column
                if 'symbol' in headers_text or 'ticker' in headers_text:
                    ticker_col = headers_text.index('symbol') if 'symbol' in headers_text else headers_text.index('ticker')
                    tickers = []
                    for row in table.find_all('tr')[1:]:  # Skip header
                        cells = row.find_all(['td', 'th'])
                        if len(cells) > ticker_col:
                            ticker = cells[ticker_col].text.strip()
                            ticker = ticker.replace('.', '-')
                            # Clean up ticker (remove any extra text)
                            if ticker:
                                # Take just the first word if there's extra text
                                ticker = ticker.split()[0] if ' ' in ticker else ticker
                                if ticker.isalpha() or '-' in ticker:
                                    tickers.append(ticker)

                    if len(tickers) >= 25:  # Dow Jones should have 30 tickers
                        logger.info(f"Fetched {len(tickers)} Dow Jones tickers from Wikipedia")
                        return tickers

    except Exception as e:
        logger.warning(f"Wikipedia Dow Jones fetch failed: {e}")

    # Fallback to hardcoded list
    logger.info("Using hardcoded Dow Jones list")
    return [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "TRV", "UNH", "V", "WMT",
    ]


def get_fallback_tickers() -> list[str]:
    """
    Fallback list of major S&P 500 stocks.
    Used when Wikipedia fetch fails.
    """
    return [
        # Technology
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "ORCL", "CRM", "ADBE",
        "AMD", "INTC", "CSCO", "IBM", "QCOM", "TXN", "AMAT", "MU", "LRCX", "SNPS",
        "NOW", "INTU", "PANW", "CDNS", "KLAC", "MRVL", "ADI", "FTNT", "CRWD", "ANET",
        # Communication Services
        "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "WBD", "TTWO",
        # Consumer Discretionary
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
        "ORLY", "AZO", "ROST", "DHI", "LEN", "MAR", "HLT", "GM", "F", "YUM",
        # Consumer Staples
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "K", "HSY", "SJM", "CAG", "KHC", "STZ", "TAP", "EL", "KR",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
        "HES", "DVN", "FANG", "HAL", "BKR", "KMI", "WMB", "OKE", "TRGP", "LNG",
        # Financials
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
        "C", "AXP", "SCHW", "CB", "MMC", "PGR", "AON", "MET", "AIG", "PRU",
        "ICE", "CME", "MCO", "TRV", "AFL", "ALL", "USB", "PNC", "TFC", "COF",
        # Healthcare
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "ISRG", "MDT", "SYK", "CVS", "CI", "ELV", "HUM", "REGN",
        "VRTX", "BSX", "ZTS", "BDX", "EW", "DXCM", "IQV", "IDXX", "MTD", "A",
        # Industrials
        "CAT", "HON", "UNP", "UPS", "RTX", "BA", "DE", "LMT", "GE", "MMM",
        "ETN", "ADP", "ITW", "EMR", "PH", "CTAS", "NSC", "CSX", "WM", "GD",
        "FDX", "NOC", "JCI", "TT", "CARR", "PCAR", "FAST", "CPRT", "ODFL", "PWR",
        # Materials
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "VMC", "MLM", "DD",
        "DOW", "PPG", "ALB", "CTVA", "CF", "MOS", "IFF", "FMC", "CE", "EMN",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR", "ARE", "SBAC", "WY", "ESS", "MAA", "UDR", "HST", "PEAK",
        # Utilities
        "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL", "ED", "PEG",
        "WEC", "ES", "AWK", "DTE", "ETR", "FE", "PPL", "AEE", "CMS", "EVRG",
    ]


def get_sp400_midcap_tickers() -> list[str]:
    """
    Fetch S&P MidCap 400 tickers from Wikipedia.
    These are mid-cap stocks - important for CANSLIM growth investing.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})

        if table is None:
            table = soup.find('table', {'class': 'wikitable'})

        tickers = []
        rows = table.find_all('tr')[1:]  # Skip header

        for row in rows:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)

        logger.info(f"Fetched {len(tickers)} S&P MidCap 400 tickers")
        return tickers

    except Exception as e:
        logger.warning(f"Could not fetch S&P MidCap 400 list: {e}")
        return get_fallback_midcap_tickers()


def get_fallback_midcap_tickers() -> list[str]:
    """Fallback list of notable mid-cap stocks."""
    return [
        # Technology
        "ACIW", "AGYS", "ASGN", "BLKB", "CACI", "CDAY", "CDW", "CHDN", "CSGS", "CSGP",
        "CVLT", "DOCU", "EPAM", "EXLS", "FFIV", "FLT", "GDDY", "GEN", "GLOB", "HQY",
        "JKHY", "MANH", "MASI", "MKSI", "NOVT", "PCTY", "PLUS", "POWI", "PWSC", "QLYS",
        "RNG", "SAIC", "SMTC", "SQSP", "TENB", "TYL", "VRSN", "WEX", "WK", "WOLF",
        # Healthcare
        "ABMD", "ACHC", "ALGN", "AMN", "BIO", "CHE", "CRL", "CTLT", "DGX", "EHC",
        "HAE", "HOLX", "HSIC", "IART", "INCY", "ITGR", "JAZZ", "LHCG", "LIVN", "MASI",
        "MEDP", "MMSI", "MOH", "NBIX", "NEO", "NHC", "NVCR", "OMCL", "PGNY", "PKI",
        "PRGO", "QDEL", "RCM", "RVMD", "SEM", "SHC", "SRPT", "STE", "TECH", "TFX",
        # Consumer
        "AAP", "AEO", "BBWI", "BJ", "BURL", "BWA", "CABO", "CAKE", "CBRL", "CCS",
        "COTY", "CROX", "DDS", "DKS", "DRI", "EAT", "ETSY", "EXPE", "FIVE", "FL",
        "GNTX", "GPC", "GPI", "HAS", "HGV", "HRB", "IPAR", "JWN", "KSS", "LAD",
        "LEA", "LKQ", "LVS", "MAN", "MLKN", "MTN", "NCLH", "NVR", "NWL", "ODP",
        # Industrials
        "AGCO", "AIT", "ALK", "ALLE", "ARNC", "ATKR", "AYI", "B", "BC", "BDC",
        "BERY", "BLD", "BLDR", "CFX", "CLH", "CMC", "CW", "DAR", "DCI", "DINO",
        "EXP", "FLS", "FLR", "GNRC", "GVA", "GWW", "HUBS", "KBR", "KEX", "LECO",
        "LII", "MAS", "MIDD", "MSA", "MTRN", "NVT", "OSK", "PLAB", "RBC", "RHI",
        # Financials
        "ALLY", "AX", "BOKF", "CADE", "CFG", "COLB", "COOP", "EWBC", "FHN", "FNB",
        "FULT", "GBCI", "HWC", "IBOC", "LSTR", "MTB", "NDAQ", "OFG", "PNFP", "PRAA",
        "RJF", "SBNY", "SEIC", "SF", "SNV", "STLD", "SYF", "TCBI", "UMBF", "VLY",
        # Energy & Materials
        "CHX", "CNX", "CVI", "DT", "HLX", "HP", "MUR", "NOV", "NTR", "OGE",
        "OGN", "OI", "OLN", "RYI", "SLB", "SM", "SUM", "TRGP", "TROX", "USG",
        # Real Estate & Utilities
        "ACC", "AVB", "BRX", "COLD", "CPT", "CUZ", "DEI", "EPR", "FR", "HR",
        "INVH", "KIM", "KRG", "LSI", "MAC", "NNN", "OHI", "OUT", "PEB", "REG",
        # Additional growth-focused mid-caps
        "ARM", "CAVA", "DUOL", "HUBS", "IOT", "KVYO", "MNDY", "ONON", "RKLB", "TOST",
    ]


def get_sp600_smallcap_tickers() -> list[str]:
    """
    Fetch S&P SmallCap 600 tickers from Wikipedia.
    Quality small-cap stocks with positive earnings requirements.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})

        if table is None:
            table = soup.find('table', {'class': 'wikitable'})

        tickers = []
        rows = table.find_all('tr')[1:]  # Skip header

        for row in rows:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)

        logger.info(f"Fetched {len(tickers)} S&P SmallCap 600 tickers")
        return tickers

    except Exception as e:
        logger.warning(f"Could not fetch S&P SmallCap 600 list: {e}")
        return get_fallback_smallcap_tickers()


def get_fallback_smallcap_tickers() -> list[str]:
    """Fallback list of notable small-cap stocks."""
    return [
        # Technology - Small Cap
        "AAOI", "ADTN", "ALRM", "APPF", "ATEN", "AVNW", "BAND", "CALX", "CASA", "CEVA",
        "CLPS", "CMBM", "COHU", "CTS", "DGII", "DIOD", "DSGX", "ELSE", "EGHT", "ETNB",
        "EVTC", "EXTR", "FARO", "FORTY", "FROG", "GSHD", "HLIT", "IIVI", "INTT", "IPGP",
        "IRTC", "KLIC", "LITE", "LSCC", "LUNA", "MAXN", "MGIC", "MGRC", "MLNK", "MXL",
        "NEOG", "OLED", "OSPN", "PAYO", "PDFS", "PI", "PLAY", "POWI", "PRFT", "PRGS",
        "PRLB", "RXT", "SANM", "SCSC", "SITM", "SMTC", "SYNA", "TTEC", "TTMI", "TUFN",
        # Healthcare - Small Cap
        "AADI", "ABCL", "ACCD", "ACHV", "ADMA", "ADPT", "AGIO", "AKRO", "ALKS", "AMPH",
        "ANAB", "ANIK", "ARNA", "ARVN", "ASTH", "ATEC", "AVNS", "AXGN", "BEAT", "BIO-B",
        "BRKR", "CARA", "CDNA", "CERS", "CHRS", "CNC", "CNMD", "CORT", "CPRI", "CPRX",
        "CUTR", "DMTK", "DVAX", "EBS", "ENTA", "ENZY", "EVH", "EXAS", "FOLD", "FTRE",
        "GH", "GMED", "HALO", "HMPT", "HRMY", "HUM", "ICUI", "IMVT", "INCY", "INSP",
        "IOVA", "IRWD", "ISEE", "ITCI", "KIDS", "KNSA", "KROS", "KRTX", "KURA", "LGND",
        # Consumer - Small Cap
        "ACCO", "ANF", "ARCO", "BCPC", "BFAM", "BGS", "BJRI", "BOOT", "CAKE", "CARS",
        "CHUY", "CMPR", "COOK", "CRAI", "CRVL", "CURV", "DAN", "DENN", "DIN", "FIZZ",
        "FLXS", "FNKO", "FOSL", "GCO", "GDEN", "GIII", "GOLF", "GPRE", "GRPN", "HBI",
        "HELE", "HIBB", "HZO", "IMKTA", "IRBT", "JACK", "JJSF", "KELYA", "KTB", "LCII",
        "LZB", "MBUU", "MCRI", "MOV", "NATH", "NGVC", "OSIS", "OTIC", "OXM", "PLAY",
        # Industrials - Small Cap
        "AAON", "ABG", "ABM", "AEIS", "AIMC", "AIR", "AIN", "AJRD", "ALEX", "ALG",
        "AMSF", "APOG", "ARCB", "ASIX", "ASTE", "ATI", "AVAV", "AVNT", "AWI", "AZZ",
        "BMI", "BRC", "BWXT", "CAI", "CBZ", "CMCO", "CMP", "CNO", "COHU", "CRS",
        "CSGS", "CW", "CXW", "DLX", "DY", "EBF", "EE", "EGP", "ELF", "ENS",
        "EPAC", "ESE", "ESNT", "EXPO", "FBIN", "FCFS", "FIX", "FWRD", "GBX", "GEF",
        # Financials - Small Cap
        "ABTX", "ACNB", "AIG", "AINV", "AJRD", "AMAL", "AMERP", "AMSF", "ANAT", "ANCX",
        "ARI", "ASB", "ATLC", "AUB", "AX", "BANF", "BANR", "BBDC", "BCBP", "BCML",
        "BFIN", "BHLB", "BHRB", "BKCC", "BKU", "BMRC", "BOCH", "BPOP", "BRKL", "BSRR",
        "BSVN", "BY", "CACC", "CADE", "CARE", "CARV", "CASH", "CBFV", "CBMB", "CBSH",
        "CCBG", "CCNE", "CFFN", "CFNB", "CHCO", "CHMG", "CIVB", "CIZN", "CLBK", "COFS",
        # Energy & Materials - Small Cap
        "AMRC", "ARCH", "AROC", "BKR", "BOOM", "BRY", "BTU", "CDEV", "CHX", "CLB",
        "CNK", "CNX", "CPE", "CTRA", "CVI", "DEN", "DNOW", "DO", "DRQ", "EGY",
        "ERII", "FET", "FLNG", "GEL", "GLNG", "GPP", "HLX", "HP", "HUN", "IOSP",
        "KOP", "KRA", "KRO", "KWR", "LBRT", "LEU", "LPG", "MARA", "MATX", "METC",
        # REITs - Small Cap
        "AAT", "ADC", "AHH", "AIRC", "AKR", "ALEX", "ALX", "APLE", "BDN", "BFS",
        "BNL", "BPYU", "BRSP", "BRT", "CBL", "CIO", "CLNC", "CLPR", "CMCT", "CPLG",
        "CSR", "CTO", "CUZ", "DEA", "DHC", "DOC", "EFC", "ELME", "EQC", "ESS",
    ]


def get_russell2000_tickers() -> list[str]:
    """
    Fetch Russell 2000 tickers from multiple sources:
    1. FMP ETF Holdings API (IWM)
    2. Yahoo Finance IWM holdings
    3. Curated fallback list

    Falls back to curated list if all fetches fail.
    """
    # Try FMP ETF Holdings API (IWM = iShares Russell 2000 ETF)
    if FMP_API_KEY:
        try:
            url = f"{FMP_BASE_URL}/etf-holder/IWM?apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                # Extract ticker symbols from holdings
                tickers = [item.get('asset') for item in data if item.get('asset')]
                if len(tickers) > 1000:  # Should be ~2000 holdings
                    logger.info(f"Fetched {len(tickers)} Russell 2000 tickers from IWM ETF (FMP)")
                    return tickers
                else:
                    logger.warning(f"FMP IWM holdings returned only {len(tickers)} tickers")
        except Exception as e:
            logger.warning(f"FMP IWM ETF holdings fetch failed: {e}")

    # Try Yahoo Finance for IWM holdings
    try:
        import yfinance as yf
        etf = yf.Ticker("IWM")

        # Try to get holdings from fund_holding_info
        holdings = None
        if hasattr(etf, 'funds_data'):
            try:
                holdings = etf.funds_data.top_holdings
            except Exception:
                pass

        # Alternative: Try institutional holders as proxy (limited but better than nothing)
        if holdings is None or holdings.empty:
            # Try to get from info dict
            info = etf.info
            # Yahoo doesn't expose full ETF holdings easily, but we can try
            pass

        if holdings is not None and not holdings.empty:
            tickers = holdings.index.tolist()
            if len(tickers) > 500:
                logger.info(f"Fetched {len(tickers)} Russell 2000 tickers from Yahoo Finance IWM")
                return tickers

    except Exception as e:
        logger.warning(f"Yahoo Finance IWM holdings fetch failed: {e}")

    # Try fetching from multiple Russell 2000 sector ETFs to build comprehensive list
    try:
        sector_etfs = get_russell2000_from_sector_etfs()
        if len(sector_etfs) > 1000:
            logger.info(f"Fetched {len(sector_etfs)} Russell 2000 tickers from sector ETFs")
            return sector_etfs
    except Exception as e:
        logger.warning(f"Sector ETF fetch failed: {e}")

    # Fallback to curated list
    logger.info("Using curated Russell 2000 list")
    return get_fallback_russell2000_tickers()


def get_russell2000_from_sector_etfs() -> list[str]:
    """
    Fetch Russell 2000 tickers by scraping multiple small-cap focused sources.
    Uses Wikipedia small-cap lists and combines with our existing data.
    """
    all_tickers = set()

    # Get all S&P SmallCap 600 tickers (overlaps with Russell 2000)
    try:
        smallcap = get_sp600_smallcap_tickers()
        all_tickers.update(smallcap)
        logger.debug(f"Added {len(smallcap)} from S&P SmallCap 600")
    except Exception:
        pass

    # Get small caps from Finviz screener (small market cap < $2B)
    try:
        finviz_tickers = get_finviz_smallcaps()
        all_tickers.update(finviz_tickers)
        logger.debug(f"Added {len(finviz_tickers)} from Finviz screener")
    except Exception:
        pass

    # Add curated Russell 2000 tickers
    curated = get_fallback_russell2000_tickers()
    all_tickers.update(curated)

    return list(all_tickers)


def get_finviz_smallcaps() -> list[str]:
    """
    Fetch small-cap tickers from Finviz screener.
    Filters: Market Cap < $2B, US exchange, has positive EPS
    """
    try:
        # Finviz screener for small-cap stocks
        url = "https://finviz.com/screener.ashx?v=111&f=cap_smallunder,exch_nasd|nyse&ft=4"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find total results count
        total_text = soup.find('td', string=lambda s: s and 'Total:' in str(s))
        if not total_text:
            return []

        # Parse first page of tickers from table
        tickers = []
        ticker_links = soup.find_all('a', {'class': 'screener-link-primary'})
        for link in ticker_links:
            ticker = link.text.strip()
            if ticker and len(ticker) <= 5:
                tickers.append(ticker)

        # Fetch additional pages (Finviz shows 20 per page)
        # We'll get first 500 tickers (25 pages) to supplement our list
        for page in range(2, 26):
            try:
                page_url = f"{url}&r={((page-1)*20)+1}"
                response = requests.get(page_url, headers=headers, timeout=10)
                if response.status_code != 200:
                    break

                soup = BeautifulSoup(response.text, 'html.parser')
                ticker_links = soup.find_all('a', {'class': 'screener-link-primary'})

                if not ticker_links:
                    break

                for link in ticker_links:
                    ticker = link.text.strip()
                    if ticker and len(ticker) <= 5:
                        tickers.append(ticker)

                # Small delay to be respectful
                import time
                time.sleep(0.2)

            except Exception:
                break

        return tickers

    except Exception as e:
        logger.warning(f"Finviz small-cap fetch failed: {e}")
        return []


def get_fallback_russell2000_tickers() -> list[str]:
    """Curated list of Russell 2000 small-cap stocks."""
    return [
        # Biotechnology & Pharmaceuticals (many trade under $20)
        "ACAD", "AGEN", "AKRO", "ALKS", "AMPH", "ARWR", "AUPH", "BBIO", "BEAM", "BCRX",
        "BHVN", "BLUE", "BPMC", "CARA", "CERS", "CLDX", "CLOV", "CMPS", "CORT", "CPRX",
        "CRNX", "CRSP", "DCPH", "DNLI", "DVAX", "DYN", "EDIT", "ENTA", "EPZM", "EXAI",
        "FATE", "FOLD", "FGEN", "GERN", "GOSS", "HALO", "HRTX", "ICPT", "IGMS", "IMVT",
        "INCY", "IONS", "IOVA", "IRWD", "ITCI", "JAZZ", "KALA", "KALV", "KPTI", "KROS",
        "KRTX", "KURA", "LGND", "LQDA", "MDGL", "MIRM", "MNKD", "MORF", "MRNS", "MRSN",
        "NBIX", "NKTR", "NTLA", "NVAX", "OCUL", "OMER", "ONCR", "ORGO", "PACB", "PCRX",
        "PDFS", "PLRX", "PRTA", "PTCT", "PGEN", "QURE", "RARE", "RCKT", "RCUS", "REGN",
        "RETA", "RGNX", "RIGL", "RVNC", "RXRX", "SAGE", "SANA", "SAVA", "SDGR", "SGEN",
        "SGMO", "SIOX", "SNDX", "SRPT", "STOK", "SUPN", "TARS", "TBIO", "TCDA", "TGTX",
        "THRM", "TPTX", "TVTX", "TWST", "UTHR", "VCNX", "VERA", "VERV", "VKTX", "VNDA",
        "VRCA", "VRTX", "XENE", "XERS", "YMAB", "ZLAB", "ZNTL", "ZYME",

        # Technology - Small Cap
        "AAOI", "AEHR", "AEIS", "AGYS", "AMBA", "AMKR", "AMSC", "AOSL", "APGE", "APPS",
        "ASGN", "ATKR", "AVNW", "BAND", "BASE", "BCOV", "BILL", "BLKB", "BMBL", "BRZE",
        "CALX", "CASA", "CDAY", "CGNX", "CHGG", "CION", "CLBT", "CLVT", "CMBM", "CNXC",
        "COHR", "COMM", "COUR", "CRUS", "CVLT", "CYBR", "DBX", "DDOG", "DIOD", "DOCN",
        "DOCU", "DOMO", "DT", "DUOL", "DV", "EGHT", "ENPH", "ENVX", "EPAM", "ESTC",
        "EVBG", "EVOP", "EXTR", "FEYE", "FIVN", "FLGT", "FORM", "FOUR", "FRSH", "FSLY",
        "FTDR", "GDYN", "GENI", "GLBE", "GLOB", "GOGO", "GTLB", "GTX", "HLIT", "HQY",
        "HUBS", "INTA", "INTT", "JAMF", "JNPR", "KLIC", "KNBE", "KTOS", "KVYO", "LITE",
        "LMND", "LPSN", "LSCC", "LUNA", "MANH", "MARA", "MAXN", "MGNI", "MKSI", "MQ",
        "MTSI", "MXL", "NCNO", "NEOG", "NET", "NOVT", "NTNX", "NVMI", "OLED", "ONTO",
        "OPCH", "OSIS", "PATH", "PAY", "PCTY", "PDFS", "PD", "PEGA", "PERI", "PI",
        "PING", "PLTK", "PLTR", "PLUS", "PMTS", "POWI", "PRFT", "PRLB", "PRGS", "PROS",
        "PSTG", "QLYS", "QTWO", "RAMP", "RDWR", "RIOT", "RMBS", "RNG", "ROKU", "RPD",
        "SAIA", "SAIL", "SAMSARA", "SCWX", "SDGR", "SEMR", "SMAR", "SMCI", "SMTC", "SNOW",
        "SPSC", "SPT", "SQSP", "SSYS", "STEP", "STNE", "SWAV", "TENB", "TER", "TNDM",
        "TOST", "TRIP", "TTWO", "TWKS", "TWLO", "UBER", "UPST", "UPWK", "VECO", "VERI",
        "VERX", "VG", "VIAV", "VNET", "VRNS", "WDAY", "WOLF", "WK", "XPEL", "YOU",
        "ZD", "ZEN", "ZI", "ZS", "ZUO",

        # Financial Services - Small Cap
        "ACGL", "AHL", "AINV", "ALLY", "AM", "ARCC", "ASB", "ATLC", "AUB", "AX",
        "BANF", "BANR", "BBDC", "BHLB", "BKU", "BPOP", "BRKL", "BXMT", "BY", "CACC",
        "CADE", "CASH", "CBSH", "CCB", "CFFN", "CFG", "CIVI", "CIVB", "CMA", "COOP",
        "CUBI", "CVBF", "CWBC", "DCOM", "ECPG", "EFSC", "ENVA", "ESGR", "ESNT", "EVTC",
        "EZPW", "FBNC", "FBP", "FCFS", "FCNCA", "FFBC", "FFIN", "FG", "FIBK", "FISI",
        "FNB", "FNCB", "FNWB", "FRME", "FSV", "FULT", "GBCI", "GNTY", "GSBC", "HAFC",
        "HBAN", "HBNC", "HCI", "HOPE", "HTBK", "HTH", "HWC", "IBCP", "IBKR", "IBOC",
        "INDB", "ISTR", "JBGS", "KREF", "LADR", "LBAI", "LC", "LCNB", "LDI", "LKFN",
        "LPRO", "MAIN", "MC", "MCB", "MCBS", "MFIN", "MFA", "MGRC", "MKTX", "MRCY",
        "MSBI", "MTB", "NBHC", "NBTB", "NCBS", "NIC", "NMIH", "NWBI", "NWFL", "NYT",
        "OFG", "ORRF", "OSBC", "OTTR", "PACW", "PATK", "PAYO", "PBCT", "PEBO", "PFBC",
        "PFS", "PNFP", "PPBI", "PRAA", "PROV", "PRU", "PTRS", "PWOD", "QCRH", "RBB",
        "RCKY", "RF", "RKT", "RM", "RNST", "RPAY", "SASR", "SBCF", "SBSI", "SFBS",
        "SFNC", "SIVB", "SLQT", "SNEX", "SNV", "SOFI", "SSBK", "SSB", "STBA", "STEL",
        "STL", "STWD", "SYF", "TBBK", "TCBI", "TCBK", "TFSL", "TRMK", "TRST", "TRUP",
        "TVTY", "TWO", "UBSI", "UCBI", "UMBF", "UMPQ", "UVSP", "VLY", "VOYA", "WABC",
        "WAFD", "WAL", "WBS", "WETF", "WFC", "WSBC", "WSFS", "XP", "ZION",

        # Consumer / Retail - Small Cap
        "AAP", "ABNB", "AEO", "ANF", "BBWI", "BIG", "BLMN", "BURL", "CAKE", "CHWY",
        "COLM", "CPRI", "CRI", "CROX", "CSS", "DBI", "DDS", "DKS", "DLTR", "DRI",
        "DXLG", "EAT", "ETSY", "EXPE", "EXPR", "EYE", "FIVE", "FL", "FNKO", "FOSL",
        "GIII", "GIL", "GME", "GMS", "GOOS", "GPS", "GRPN", "HIBB", "HVT", "IRBT",
        "JILL", "JWN", "KSS", "LCUT", "LE", "LEG", "LESL", "LEVI", "LULU", "LZB",
        "M", "MOV", "MSGS", "NATH", "NGVC", "OBIC", "ODP", "OXM", "PETS", "PII",
        "PLAY", "PLCE", "PRPL", "PVH", "REAL", "RENT", "RH", "RVLV", "SHOO", "SIG",
        "SKX", "SNBR", "SPWH", "STKS", "STOR", "TCS", "TDUP", "TGT", "TXRH", "ULTA",
        "URBN", "VFC", "VNCE", "W", "WEBR", "WING", "WISH", "WRBY", "WSM", "WTRG",
        "WWW", "YETI", "ZG", "ZUMZ",

        # Industrial - Small Cap
        "AAON", "AAXN", "ABG", "AGCO", "AIN", "AIRC", "AJRD", "ALG", "ALSN", "AMED",
        "AME", "ARCB", "AROC", "ASPN", "ATI", "AVAV", "AXON", "AYI", "B", "BC",
        "BERY", "BLD", "BLDR", "BMI", "BWXT", "CAR", "CBT", "CECO", "CHE", "CMCO",
        "CMC", "CMP", "CNH", "COHU", "CR", "CRS", "CSL", "CW", "CXW", "DAN",
        "DCI", "DLX", "DNOW", "DY", "EAF", "EE", "EFOI", "EGP", "ENS", "EPAC",
        "ESE", "EXP", "EXPO", "FBIN", "FCPT", "FIX", "FLR", "FSTR", "GBX", "GEF",
        "GEO", "GFF", "GGG", "GHC", "GHL", "GKOS", "GMS", "GNRC", "GNW", "GNTX",
        "GRC", "GVA", "GWW", "HDS", "HI", "HIW", "HNI", "HRI", "HY", "IESC",
        "IEX", "ILMN", "INST", "JBHT", "JBL", "JBSS", "KAI", "KBR", "KEX", "KMT",
        "KNX", "LECO", "LII", "LNN", "LSTR", "MATX", "MBC", "MDU", "MHK", "MIDD",
        "MLI", "MLM", "MOG-A", "MSA", "MSM", "MWA", "NDSN", "NEU", "NJR", "NNI",
        "NPO", "NVR", "NYT", "OC", "OGE", "OSK", "OTIS", "PATK", "PECO", "PH",
        "PHM", "PLAB", "PNR", "POR", "POWW", "PRIM", "PSN", "PTC", "R", "RBC",
        "RCM", "RDN", "REXR", "RHI", "RMD", "RNR", "ROL", "RRX", "RS", "RXO",
        "SANM", "SBUX", "SEIC", "SEM", "SFM", "SHO", "SKY", "SNA", "SNDR", "SNX",
        "SPB", "SPXC", "SR", "SSD", "SUM", "SWK", "SWN", "SXI", "TEX", "TFX",
        "TGH", "THS", "TKR", "TMHC", "TNC", "TPH", "TREX", "TRN", "TSN", "TTC",
        "TTEK", "TTMI", "TUP", "UFPI", "UHAL", "UNF", "URI", "USPH", "UTL", "VCEL",
        "VNT", "VREX", "VSH", "WAB", "WERN", "WLK", "WMS", "WOR", "WTS", "WWD",
        "XYL",

        # Energy - Small Cap
        "AM", "APA", "AR", "AROC", "ARIS", "ARNC", "BCEI", "BKV", "BORR", "BRY",
        "BTU", "CDEV", "CHRD", "CHX", "CIVI", "CLB", "CLR", "CNX", "COP", "CPE",
        "CTRA", "CVI", "CXO", "DEN", "DINO", "DNR", "DO", "DRQ", "ERF", "EXE",
        "FANG", "FET", "FTI", "GEVO", "GPOR", "GPRE", "HEP", "HLX", "HP", "HPK",
        "KOS", "LEU", "LBRT", "LPI", "MGY", "MPC", "MPLN", "MRO", "MTDR", "MUR",
        "NBR", "NEX", "NOG", "NOV", "OAS", "OII", "OIS", "OVV", "PARR", "PDCE",
        "PDS", "PTEN", "PXD", "QEP", "RES", "REX", "RIG", "ROCC", "RRC", "SDRL",
        "SGU", "SLCA", "SM", "SUN", "SWN", "TDW", "TELL", "TGS", "TPIC", "TRMD",
        "TRP", "USAC", "VAL", "VNOM", "VVV", "WHD", "WPX", "WTI", "XEC",

        # Healthcare - Small Cap (non-biotech)
        "AADI", "ACHC", "ADUS", "AHCO", "ALIT", "AMN", "AMWD", "APLS", "ARNA", "ASTH",
        "ATRC", "AVNS", "AXGN", "AZTA", "BIO", "BLFS", "BVS", "CCRN", "CHE", "CHRS",
        "CNC", "CNMD", "CRVL", "CTLT", "CYH", "DXCM", "ENSG", "EHC", "EVH", "EXAS",
        "FTRE", "GH", "GMED", "HAE", "HCA", "HCAT", "HLNE", "HQY", "HSIC", "IART",
        "INGN", "INMD", "INSP", "ISRG", "KIDS", "KNSA", "KRYS", "LHCG", "LH", "LIVN",
        "LNTH", "MD", "MDRX", "MEDP", "MOH", "MYGN", "NEOG", "NHC", "NUVA", "NVRO",
        "OFIX", "OMCL", "OPCH", "OSH", "PBH", "PDCO", "PENN", "PHAS", "PHG", "PHR",
        "PINC", "PKI", "PNTG", "PRGO", "PRVA", "QDEL", "QGEN", "RCM", "RDNT", "RMD",
        "RVMD", "SEM", "SHC", "SGRY", "SIBN", "SLP", "STE", "STRL", "SURG", "SYNA",
        "SYK", "TCON", "TFX", "THC", "TTEC", "USPH", "VAR", "VCYT", "VEEV", "WST",
        "XRAY",

        # Materials - Small Cap
        "AIMC", "AKUS", "ARCH", "ASH", "ASIX", "AVTR", "AWI", "AXTA", "BCPC", "BCC",
        "BLD", "BMS", "BPOP", "BRKR", "CBT", "CC", "CLF", "CLW", "CMC", "CMP",
        "CRS", "CSWI", "CTVA", "CVI", "CYT", "DDD", "DNMR", "DOOR", "ENV", "ESI",
        "FUL", "GCP", "GLATF", "GNK", "GPK", "GRBK", "GTLS", "GWR", "HBB", "HUN",
        "HWKN", "IBP", "IIIN", "IOSP", "KALU", "KMT", "KNF", "KOP", "KRA", "KRO",
        "KWR", "LAD", "LAUR", "LDL", "LGIH", "LTHM", "LXU", "MAT", "MATW", "MG",
        "MTRN", "NEU", "NPK", "NUE", "OI", "OLN", "OMI", "PAG", "PAGS", "PATK",
        "PBF", "PCT", "PKG", "POOL", "PPC", "RCUS", "RPM", "RYI", "RYAM", "SCL",
        "SEE", "SITC", "SLG", "SLGN", "SMPL", "SMLP", "SON", "SSNC", "STLD", "STNG",
        "SUM", "SUP", "SWM", "SXT", "TECK", "TMQ", "TPL", "TRNO", "UFPI", "USG",
        "USLM", "USM", "VCEL", "VHI", "VMC", "VSTO", "WFG", "WLK", "WMS", "WRK",
        "X", "ZEUS", "ZTR",

        # REITs - Small Cap
        "AAT", "ACC", "ADC", "AFCG", "AHH", "AIRC", "AIV", "AKR", "ALEX", "ALX",
        "AMTD", "APLE", "ARE", "BDN", "BFS", "BNL", "BRG", "BRSP", "BRT", "BRX",
        "BXMT", "BXMT", "CBL", "CDP", "CHCT", "CLNC", "CLPR", "CLP", "CMCT", "CONE",
        "COLD", "CPLG", "CSR", "CTO", "CUZ", "DEA", "DEI", "DHC", "DOC", "EARN",
        "EFC", "EGBN", "ELME", "EPRT", "EQC", "EQR", "ESS", "FCPT", "FPI", "FR",
        "GMRE", "GOOD", "GTY", "HIW", "HRTG", "HT", "INN", "IIPR", "ILPT", "INVH",
        "IRET", "IRM", "IRT", "JBGS", "KREF", "KRG", "LADR", "LAND", "LMRK", "LTC",
        "LXP", "MAC", "MAA", "MFA", "MGP", "MPW", "NAHI", "NHI", "NLY", "NNN",
        "NSA", "NXRT", "NYC", "O", "OFC", "OHI", "OLP", "OPI", "OUT", "PECO",
        "PEB", "PEI", "PGRE", "PK", "PLYM", "PSTL", "REXR", "RHP", "RITM", "RLJ",
        "ROIC", "RPT", "RVI", "RWT", "SAFE", "SBAC", "SBRA", "SHO", "SLG", "SMTA",
        "SPG", "SRC", "SREA", "STAR", "STAG", "STOR", "SUI", "SVC", "TCO", "TRNO",
        "TRTX", "TWO", "UBA", "UDR", "UE", "UHT", "UMH", "UNIT", "VICI", "VNO",
        "VRE", "VTR", "WPC", "WPG", "WRI", "WSR", "XHR",

        # Additional small/micro caps frequently in portfolios
        "ZETA", "HUMA", "LCTX", "ONDS", "NANC",  # User portfolio stocks
        "LUNR", "RKLB", "IONQ", "RGTI", "QBTS",  # Space/quantum computing
        "PLTR", "SNOW", "NET", "CRWD", "ZS",  # High-growth tech
        "RIVN", "LCID", "FSR", "NKLA", "GOEV",  # EV
        "SOFI", "AFRM", "UPST", "HOOD", "COIN",  # Fintech
        "DKNG", "PENN", "RSI", "GENI", "BETZ",  # Gaming/sports betting
    ]


def _load_env():
    """Load .env file for local testing."""
    global FMP_API_KEY
    from pathlib import Path
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ.setdefault(key, val)
        FMP_API_KEY = os.environ.get('FMP_API_KEY', '')


if __name__ == "__main__":
    _load_env()

    print("Fetching ticker lists...")
    print(f"FMP API Key: {'configured' if FMP_API_KEY else 'NOT SET'}\n")

    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    dowjones = get_dowjones_tickers()
    midcap400 = get_sp400_midcap_tickers()
    smallcap600 = get_sp600_smallcap_tickers()
    russell = get_russell2000_tickers()
    portfolio = get_portfolio_tickers()
    all_tickers = get_all_tickers()

    print(f"\nIndex Breakdown:")
    print(f"  S&P 500:          {len(sp500):>5} tickers")
    print(f"  Nasdaq 100:       {len(nasdaq100):>5} tickers")
    print(f"  Dow Jones 30:     {len(dowjones):>5} tickers")
    print(f"  S&P MidCap 400:   {len(midcap400):>5} tickers")
    print(f"  S&P SmallCap 600: {len(smallcap600):>5} tickers")
    print(f"  Russell 2000:     {len(russell):>5} tickers")
    print(f"  Portfolio:        {len(portfolio):>5} tickers")
    print(f"\n  Combined (deduplicated): {len(all_tickers)} tickers")

    # Check for missing portfolio tickers
    all_set = set(all_tickers)
    missing = [t for t in portfolio if t not in all_set]
    if missing:
        print(f"\nWARNING: Portfolio tickers not in combined list: {missing}")
    else:
        print(f"\n✓ All portfolio tickers included in scan")
