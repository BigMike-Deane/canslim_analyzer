#!/usr/bin/env python3
"""
Fidelity CSV Upload Automation Script

Watches a folder for new Fidelity CSV exports and automatically uploads them
to the CANSLIM Analyzer API.

Usage:
    # One-shot mode: Upload specific files
    python fidelity_upload.py --positions "Portfolio_Positions_*.csv"
    python fidelity_upload.py --activity "Accounts_History*.csv"

    # Watch mode: Monitor Downloads folder for new CSVs
    python fidelity_upload.py --watch

    # Upload both from a folder
    python fidelity_upload.py --folder "C:/Users/bayer/Downloads"

Configuration:
    Set CANSLIM_API_URL environment variable or edit the default below.
    Default: http://100.104.189.36:8001 (Tailscale VPS)
"""

import os
import sys
import glob
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# API endpoint - configurable via environment variable
API_URL = os.environ.get('CANSLIM_API_URL', 'http://100.104.189.36:8001')


def upload_positions(filepath: str) -> dict:
    """Upload a Fidelity Positions CSV to the API."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None

    print(f"  Uploading positions: {filepath.name}")
    with open(filepath, 'rb') as f:
        response = requests.post(
            f'{API_URL}/api/fidelity/upload-positions',
            files={'file': (filepath.name, f, 'text/csv')},
            timeout=30,
        )

    if response.ok:
        result = response.json()
        print(f"  OK: {result['positions_count']} positions, "
              f"${result['total_value']:,.2f} total, "
              f"${result['cash_balance']:,.2f} cash")
        if result.get('parse_errors'):
            for err in result['parse_errors']:
                print(f"  WARN: {err}")
        return result
    else:
        print(f"  ERROR {response.status_code}: {response.text}")
        return None


def upload_activity(filepath: str) -> dict:
    """Upload a Fidelity Activity CSV to the API."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None

    print(f"  Uploading activity: {filepath.name}")
    with open(filepath, 'rb') as f:
        response = requests.post(
            f'{API_URL}/api/fidelity/upload-activity',
            files={'file': (filepath.name, f, 'text/csv')},
            timeout=30,
        )

    if response.ok:
        result = response.json()
        print(f"  OK: {result['new_trades']} new trades, "
              f"{result['skipped_duplicates']} duplicates skipped, "
              f"{result['dividends_found']} dividends found")
        if result.get('parse_errors'):
            for err in result['parse_errors']:
                print(f"  WARN: {err}")
        return result
    else:
        print(f"  ERROR {response.status_code}: {response.text}")
        return None


def sync_to_portfolio() -> dict:
    """Sync Fidelity positions to the manual portfolio."""
    print("  Syncing to portfolio...")
    response = requests.post(f'{API_URL}/api/fidelity/sync-to-portfolio', timeout=30)
    if response.ok:
        result = response.json()
        print(f"  OK: {result['added']} added, {result['updated']} updated, "
              f"{result['removed']} removed")
        return result
    else:
        print(f"  ERROR {response.status_code}: {response.text}")
        return None


def find_latest_file(folder: str, pattern: str) -> str:
    """Find the most recently modified file matching a pattern."""
    folder = Path(folder)
    matches = list(folder.glob(pattern))
    if not matches:
        return None
    return str(max(matches, key=lambda p: p.stat().st_mtime))


def upload_from_folder(folder: str, sync: bool = False):
    """Find and upload the latest Fidelity CSVs from a folder."""
    folder = Path(folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    print(f"\nScanning: {folder}")
    print(f"API: {API_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # Find latest positions file
    positions_file = find_latest_file(folder, "Portfolio_Positions_*.csv")
    if positions_file:
        upload_positions(positions_file)
    else:
        print("  No positions CSV found (Portfolio_Positions_*.csv)")

    # Find latest activity file
    activity_file = find_latest_file(folder, "Accounts_History*.csv")
    if activity_file:
        upload_activity(activity_file)
    else:
        print("  No activity CSV found (Accounts_History*.csv)")

    if sync and positions_file:
        sync_to_portfolio()

    print("-" * 50)
    print("Done.\n")


def watch_folder(folder: str, interval: int = 30, sync: bool = False):
    """
    Watch a folder for new Fidelity CSV files and upload when detected.
    Polls every `interval` seconds.
    """
    folder = Path(folder)
    print(f"\nWatching: {folder}")
    print(f"API: {API_URL}")
    print(f"Poll interval: {interval}s")
    print(f"Auto-sync: {'yes' if sync else 'no'}")
    print("Press Ctrl+C to stop.\n")

    seen_files = set()

    # Mark existing files as already seen
    for pattern in ["Portfolio_Positions_*.csv", "Accounts_History*.csv"]:
        for f in folder.glob(pattern):
            seen_files.add(str(f))

    print(f"Tracking {len(seen_files)} existing files. Waiting for new ones...\n")

    try:
        while True:
            time.sleep(interval)

            for pattern, uploader in [
                ("Portfolio_Positions_*.csv", upload_positions),
                ("Accounts_History*.csv", upload_activity),
            ]:
                for f in folder.glob(pattern):
                    fpath = str(f)
                    if fpath not in seen_files:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New file detected!")
                        result = uploader(fpath)
                        seen_files.add(fpath)

                        if sync and result and "Portfolio_Positions" in fpath:
                            sync_to_portfolio()

    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Fidelity CSV exports to CANSLIM Analyzer"
    )
    parser.add_argument('--positions', help='Path to positions CSV file')
    parser.add_argument('--activity', help='Path to activity CSV file')
    parser.add_argument('--folder', help='Folder to scan for CSVs')
    parser.add_argument('--watch', action='store_true', help='Watch folder for new files')
    parser.add_argument('--interval', type=int, default=30, help='Watch poll interval (seconds)')
    parser.add_argument('--sync', action='store_true', help='Auto-sync positions to portfolio')
    parser.add_argument('--api', help=f'API URL (default: {API_URL})')

    args = parser.parse_args()

    if args.api:
        global API_URL
        API_URL = args.api

    # Determine default folder
    default_folder = Path.home() / "Downloads"

    if args.positions:
        upload_positions(args.positions)
        if args.sync:
            sync_to_portfolio()
    elif args.activity:
        upload_activity(args.activity)
    elif args.watch:
        folder = args.folder or str(default_folder)
        watch_folder(folder, args.interval, args.sync)
    elif args.folder:
        upload_from_folder(args.folder, args.sync)
    else:
        # Default: scan Downloads folder
        upload_from_folder(str(default_folder), args.sync)


if __name__ == '__main__':
    main()
