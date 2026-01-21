"""
Stock Ticker List Module
Includes S&P 500, S&P MidCap 400, S&P SmallCap 600, and Russell 2000 stocks.
Also fetches portfolio tickers to ensure they're always scanned.
"""

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


def get_all_tickers(include_portfolio: bool = True) -> list[str]:
    """
    Get combined list of all major index tickers plus portfolio holdings.

    Includes:
    - S&P 500 (large cap)
    - S&P MidCap 400 (mid cap)
    - S&P SmallCap 600 (small cap)
    - Russell 2000 (small cap, curated list)
    - Portfolio tickers (always scanned)
    """
    sp500 = get_sp500_tickers()
    midcap400 = get_sp400_midcap_tickers()
    smallcap600 = get_sp600_smallcap_tickers()
    russell = get_russell2000_tickers()

    # Start with portfolio tickers (highest priority)
    combined = []
    if include_portfolio:
        portfolio = get_portfolio_tickers()
        combined.extend(portfolio)

    # Add index tickers
    combined.extend(sp500)
    combined.extend(midcap400)
    combined.extend(smallcap600)
    combined.extend(russell)

    # Remove duplicates while preserving order (portfolio first)
    seen = set()
    unique = []
    for ticker in combined:
        if ticker not in seen:
            seen.add(ticker)
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
        positions = db.query(PortfolioPosition.ticker).distinct().all()
        tickers = [p.ticker for p in positions]
        db.close()

        if tickers:
            logger.info(f"Loaded {len(tickers)} portfolio tickers from database")
            return tickers
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
    Fetch S&P 500 tickers from Wikipedia.
    Falls back to a curated list if fetch fails.
    """
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
                # Clean up ticker (remove any notes)
                ticker = ticker.replace('.', '-')  # BRK.B -> BRK-B for yfinance
                tickers.append(ticker)

        return tickers

    except Exception as e:
        print(f"Warning: Could not fetch S&P 500 list from Wikipedia: {e}")
        print("Using fallback list of major stocks...")
        return get_fallback_tickers()


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
    Curated list of Russell 2000 small-cap stocks.
    Focus on liquid, actively traded small caps.
    """
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


if __name__ == "__main__":
    print("Fetching ticker lists...")
    sp500 = get_sp500_tickers()
    midcap400 = get_sp400_midcap_tickers()
    smallcap600 = get_sp600_smallcap_tickers()
    russell = get_russell2000_tickers()
    portfolio = get_portfolio_tickers()
    all_tickers = get_all_tickers()

    print(f"\nIndex Breakdown:")
    print(f"  S&P 500:        {len(sp500):>4} tickers")
    print(f"  S&P MidCap 400: {len(midcap400):>4} tickers")
    print(f"  S&P SmallCap 600: {len(smallcap600):>4} tickers")
    print(f"  Russell 2000:   {len(russell):>4} tickers (curated)")
    print(f"  Portfolio:      {len(portfolio):>4} tickers")
    print(f"\nCombined (deduplicated): {len(all_tickers)} tickers")

    # Check for missing portfolio tickers
    all_set = set(all_tickers)
    missing = [t for t in portfolio if t not in all_set]
    if missing:
        print(f"\nWARNING: Portfolio tickers not in combined list: {missing}")
