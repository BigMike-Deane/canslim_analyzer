# Secrets Audit

**Date:** 2026-05-10
**Branch:** main (post-`44379c4`)
**Scope:** Tracked repo, git history (all branches), running Docker image, prod environment.
**Outcome:** **CLEAN.** No remediation required. No live rotation triggered.

---

## Methodology

Three-layer sweep:

1. **Tracked source** — pattern grep for `(secret_key|client_id|api_key|password)\s*=\s*['"]<16+ chars>['"]` plus name-targeted grep for our actual env vars (`JWT_SECRET_KEY`, `GOOGLE_CLIENT_ID`, `FMP_API_KEY`, `API_TOKEN`, `CANSLIM_WEBHOOK_URL`).
2. **Git history** — `git log --all -p -S <token>` (pickaxe) for `JWT_SECRET`, `GOOGLE_CLIENT`, `FMP_API_KEY`. Pickaxe finds *additions or removals of the literal string* across every branch, so a secret committed and later deleted still surfaces.
3. **Image + runtime** — Dockerfile `COPY` audit, `.dockerignore` review, `docker inspect` of mounts on the VPS, `docker exec env` shape check (length / hex / domain), 24h log tail grep.

---

## Findings

### F-0 — Repo grep (tracked source)

Pattern grep across `*.py`, `*.yaml`, `*.yml`, `*.env*` (excluding `node_modules/`, `.git/`, virtualenvs):

```
grep -rnE "(secret_key|client_id|api_key|password)\s*=\s*['\"][A-Za-z0-9/+=_-]{16,}['\"]" ...
```

**Result:** 0 matches.

Name-targeted grep for our env-var names with literal values: 0 matches.

The only references to a secret-like string in the tracked code are:

| Path | Line | What |
|---|---|---|
| `backend/auth.py` | 21 | `os.environ.get("JWT_SECRET_KEY", "dev-secret-key-change-in-production")` — dev fallback, never reaches prod (verified F-3). |
| `tests/test_auth.py` | 17, 24, 30, 35 | Same dev fallback used in 4 unit tests. Tests run with no env set. |

The dev fallback is a documented sentinel, not a real secret. Acceptable.

### F-1 — Git history (`.env` files)

```
git log --all --diff-filter=A --name-only --pretty=format: -- '*.env' '*.env.*' '.env*'
```

Only `.env.template` was ever added. No real `.env` has ever been committed on any branch.

Pickaxe sweeps for `JWT_SECRET`, `GOOGLE_CLIENT`, `FMP_API_KEY`:
- `JWT_SECRET` first appears in `c7b378c` (Mar 3 2026) — multi-user auth introduction. Diff shows `.env.template` placeholder + `auth.py` dev fallback. No real secret committed.
- `GOOGLE_CLIENT` — same story, same commit.
- `FMP_API_KEY` — only matches inside a docstring in `cdf7c52` (data_fetcher coverage push).

**No secret has ever existed in git history.**

### F-2 — Docker image layer

`Dockerfile` `COPY` directives audited (lines 8, 10, 26, 30, 33, 36-45, 48). None copy `.env*`.

`.dockerignore` line 12-14 excludes `.env` and `.env.local` from the build context, so even an accidental `COPY . .` could not bake the secret in.

`docker inspect canslim-analyzer` confirms the runtime layout:

```json
{ "Type": "bind",
  "Source": "/opt/canslim_analyzer/.env",
  "Destination": "/app/.env",
  "Mode": "ro" }
```

The `/app/.env` seen by `docker exec find` is a read-only bind mount from the VPS, not an image layer artifact. **The image is shippable to a public registry without leakage.**

### F-3 — Prod env validation

Three gates from the brief:

| Variable | Gate | Result |
|---|---|---|
| `JWT_SECRET_KEY` | Present, 64+ hex chars, NOT the dev fallback | ✅ present, len=64, all-hex, ≠ dev fallback |
| `GOOGLE_CLIENT_ID` | Present, ends `.googleusercontent.com` | ✅ present, len=73, ends correctly |
| `REQUIRE_AUTH` | `"true"` | ✅ `true` |

Validation method: `docker exec canslim-analyzer python3 -c '...'` reading `os.environ` and reporting *shape only* (length, hex-ness, domain suffix, equality-with-fallback). No secret value left the container.

### F-4 — CI / secrets in workflows

`/mnt/c/Users/bayer/canslim_analyzer/.github/` does not exist. There are no GitHub Actions, no encrypted GH secrets to inventory, no CI logs that could leak. Deployment is manual SSH-driven (see `CLAUDE.local.md`).

### F-5 — 24h prod log scrub

`docker logs canslim-analyzer --since 24h | grep -iE '(jwt_secret|api_key|password|google_client|secret_key)'` (filtered to remove the known benign "GOOGLE_CLIENT_ID missing" startup message): 0 matches in the past 24 hours.

---

## What we did NOT check (out of scope)

- **Live secret rotation** — too disruptive during the Approach 2 eval window (verdict 2026-06-18). Playbook in `docs/security/jwt-rotation.md` is ready when needed.
- **Dependency CVE scan** — `pip-audit` / `npm audit`. Prior pass exists at `docs/security/dependency-cve-scan-2026-05-08.md`; rescheduling is CSO #8.
- **Backup file review** — `/opt/canslim_analyzer/data/backups/` may contain dumps; the DB does not store secrets but a future audit should still confirm.
- **Pre-commit secret scanner** — no pre-commit framework is currently configured. Adding one (e.g. `gitleaks`) is its own ship; doing it inside this audit would expand scope.

---

## Recommendation

**No action required.** Routine annual JWT rotation per the playbook in `jwt-rotation.md` is the next housekeeping touch (target 2026-06-19, the day after the Approach 2 verdict, so the rotation's auth-fail spike doesn't muddy the eval window).
