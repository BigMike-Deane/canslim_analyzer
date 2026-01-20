"""
Stock Ticker List Module
Includes S&P 500 and Russell 2000 small-cap stocks
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_all_tickers() -> list[str]:
    """
    Get combined list of S&P 500 and Russell 2000 tickers.
    """
    sp500 = get_sp500_tickers()
    russell = get_russell2000_tickers()
    # Combine and remove duplicates
    combined = list(dict.fromkeys(sp500 + russell))
    return combined


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
    ]


if __name__ == "__main__":
    sp500 = get_sp500_tickers()
    russell = get_russell2000_tickers()
    all_tickers = get_all_tickers()
    print(f"S&P 500: {len(sp500)} tickers")
    print(f"Russell 2000: {len(russell)} tickers")
    print(f"Combined (deduplicated): {len(all_tickers)} tickers")
