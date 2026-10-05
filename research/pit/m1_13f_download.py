"""M1: download SEC Form 13F data sets (institutional holdings, 2013Q2->) and
fails-to-deliver files (CUSIP <-> ticker-at-the-time, the 13F join key).

Sequential, resumable (skips files already on disk), SEC-compliant UA from .env.
"""
import re
import time
from pathlib import Path

import requests

from common import SEC_DIR, SEC_HEADERS

BASE = "https://www.sec.gov"
F13_DIR, FTD_DIR = SEC_DIR / "13f", SEC_DIR / "ftd"
for d in (F13_DIR, FTD_DIR):
    d.mkdir(parents=True, exist_ok=True)


def download(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    for attempt in range(4):
        r = requests.get(url, headers=SEC_HEADERS, timeout=600, stream=True)
        if r.status_code == 404:
            return "404"
        if r.ok:
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            time.sleep(0.5)
            return "ok"
        time.sleep(10 * 2 ** attempt)
    return f"failed {r.status_code}"


def main():
    page = requests.get(f"{BASE}/data-research/sec-markets-data/form-13f-data-sets",
                        headers=SEC_HEADERS, timeout=60).text
    links = sorted(set(re.findall(r'href="([^"]*form13f\.zip)"', page)))
    print(f"13F data sets: {len(links)}", flush=True)
    for href in links:
        print(f"  {href.rsplit('/', 1)[-1]}: {download(BASE + href, F13_DIR / href.rsplit('/', 1)[-1])}", flush=True)

    # Fails-to-deliver: semi-monthly cnsfailsYYYYMM{a,b}.zip, links from SEC's page.
    page = requests.get(f"{BASE}/data-research/sec-markets-data/fails-deliver-data",
                        headers=SEC_HEADERS, timeout=60).text
    ftd = sorted(set(h for h in re.findall(r'href="([^"]*cnsfails(\d{6})[ab]\.zip)"', page)
                     if h[1] >= "201301"))
    print(f"FTD files listed (2013->): {len(ftd)}", flush=True)
    n = 0
    for href, _ in ftd:
        name = href.rsplit("/", 1)[-1]
        status = download(BASE + href, FTD_DIR / name)
        n += status in ("ok", "cached")
        time.sleep(0.3)
    print(f"FTD files on disk: {n}", flush=True)


if __name__ == "__main__":
    main()
