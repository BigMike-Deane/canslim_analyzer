"""Mint the read-only tooling token (see backend/auth.py READONLY_*).

    docker exec canslim-analyzer python3 -m backend.mint_readonly_token [--days 180]

Prints ONLY the token on stdout, so the caller can redirect it straight
into a file without it ever passing through a terminal or a chat. Minting
again replaces (revokes) the previous token.
"""

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--user-id", type=int, default=1)
    args = ap.parse_args(argv)
    from backend.auth import create_readonly_token
    token = create_readonly_token(user_id=args.user_id, days=args.days)
    print(token)
    print(f"read-only token minted for user {args.user_id}, valid {args.days} days; "
          "any previous read-only token is now revoked", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
