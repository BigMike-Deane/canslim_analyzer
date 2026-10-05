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

    # Fails-to-deliver: cnsfailsYYYYMM{a,b}.zip, semi-monthly.
    n = 0
    for y in range(2013, 2027):
        for m in range(1, 13):
            for half in "ab":
                name = f"cnsfails{y}{m:02d}{half}.zip"
                status = download(f"{BASE}/files/data/fails-deliver-data/{name}", FTD_DIR / name)
                n += status in ("ok", "cached")
    print(f"FTD files on disk: {n}", flush=True)


if __name__ == "__main__":
    main()
