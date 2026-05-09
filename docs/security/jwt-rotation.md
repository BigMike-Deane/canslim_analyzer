# JWT Secret Rotation Playbook

**Last reviewed:** 2026-05-10
**Owner:** Repo admin (single-user prod ops)
**Time to execute:** ~15 minutes hands-on + 7-day grace window
**Audit log:** Rotation produces a brief, bounded spike of `auth_fail` / `invalid_token` rows in the in-memory audit log shipped in CSO #6 (commit `44379c4`). Use that to confirm the rotation is healthy.

---

## When to rotate

Rotate `JWT_SECRET_KEY` when any of these fire:

- **Leak suspected.** Secret appeared in a log file, screenshot, paste, dependency error trace, or CI output.
- **Dependency CVE.** A `python-jose` / `cryptography` advisory lands that affects HS256 token integrity.
- **Personnel change.** A trusted operator who handled `.env` is offboarded.
- **Annual hygiene.** Default cadence: every 12 months. Schedule the next routine rotation for 2026-06-19 (one day after the Approach 2 verdict on 2026-06-18 — do **not** rotate during an A/B-eval window because the auth-fail spike pollutes the eval).

Do **not** rotate impulsively during:
- An open A/B-eval window (Approach 2 currently runs through 2026-06-18).
- A market-hours scanner cycle when traders are actively syncing portfolios.

---

## What rotation invalidates (the constraint)

Both access and refresh tokens are HS256-signed with `SECRET_KEY` (`backend/auth.py:65, 72`). A naive flip of the env var instantly invalidates:

- **All live access tokens** (TTL: 30 minutes).
- **All live refresh tokens** (TTL: 7 days).

This means every browser session is logged out and every mobile PWA loses its refresh chain in a single moment. We avoid that by running both keys side-by-side for 7 days.

---

## Grace-period rotation procedure

We use a **dual-key verify, single-key mint** pattern: the new key becomes active for minting immediately; the previous key stays alive for verification only, until the refresh-token TTL has rolled over every live session.

Convention: the previous key is named `JWT_SECRET_KEY_PREV`. (The original brief called it `_NEXT`; `_PREV` is more honest because it tracks the key being phased out.)

### Step 1 — Generate the new secret

```bash
NEW=$(openssl rand -hex 32)
echo "$NEW"   # 64 hex chars; eyeball-check before pasting
```

Use `openssl rand -hex 32`, not `head /dev/urandom | base64`, because the existing prod key is 64-hex (verified in `secrets-audit-2026-05-10.md`) and consistency makes future audits one-shot.

### Step 2 — Stage the previous secret on the VPS

SSH to `100.104.189.36` and edit `/opt/canslim_analyzer/.env`:

```ini
# Before rotation
JWT_SECRET_KEY=<old-64-hex>

# After Step 2 — same file, two keys
JWT_SECRET_KEY=<NEW>           # promoted: active for minting
JWT_SECRET_KEY_PREV=<old>      # verify-only fallback, removed after Step 5
```

Do **not** redeploy yet. The new env var is loaded on next container start.

### Step 3 — Patch `backend/auth.py` and `backend/routes/auth.py`

The dual-key code lives in two decode sites and zero encode sites (mint stays single-key — the new key).

**`backend/auth.py:21`** — load both keys:

```python
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
SECRET_KEY_PREV = os.environ.get("JWT_SECRET_KEY_PREV", "")  # empty when not rotating
```

**`backend/auth.py:126`** — access-token verify, dual-key:

```python
try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
except JWTError:
    if SECRET_KEY_PREV:
        try:
            payload = jwt.decode(token, SECRET_KEY_PREV, algorithms=[ALGORITHM])
        except JWTError:
            raise HTTPException(...)  # existing 401 path
    else:
        raise HTTPException(...)
```

**`backend/routes/auth.py:93`** — refresh-token verify, identical pattern.

Encode sites (`auth.py:65`, `auth.py:72`) stay as-is. New tokens are minted with the new `SECRET_KEY`.

### Step 4 — Deploy

```bash
ssh root@100.104.189.36 'cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build'
```

The moment the new container comes up:
- Existing access tokens (signed with old key) → fall through to `SECRET_KEY_PREV` and verify ✅
- Existing refresh tokens (signed with old key) → same ✅
- New logins → mint with new `SECRET_KEY` ✅
- A new login by an attacker who held only the old leaked key → cannot mint (no token endpoint accepts arbitrary keys) ✅

### Step 5 — Wait out the grace period

Wait **7 days** (refresh-token TTL). After 7d, every live refresh token has either expired or been used to mint a fresh access token, so all live tokens are signed with the new `SECRET_KEY`.

Faster alternative: force-logout via `is_active = False` toggle on every user, then `True`. This invalidates all sessions immediately and obviates the grace period — but every operator has to log back in within the hour, which is more disruptive than 7 days of dual-key verify. Use only if a leak is confirmed and active exploitation is observed.

### Step 6 — Remove the fallback

Edit `/opt/canslim_analyzer/.env`: delete the `JWT_SECRET_KEY_PREV` line.

Revert the dual-key block in `backend/auth.py:126` and `backend/routes/auth.py:93` to single-key. Keep the line `SECRET_KEY_PREV = os.environ.get(...)` if you anticipate another rotation soon, otherwise delete it too.

Redeploy. Grace window is closed.

---

## Audit-log signature: healthy vs unhealthy rotation

The CSO #6 audit log (`backend/audit_log.py`, in-memory ring buffer reachable via `GET /api/admin/audit-log`) records every `auth_fail` / `invalid_token` event with timestamp + source_ip.

**Healthy rotation looks like:**

| Window | Expected pattern |
|---|---|
| 0–5 min after Step 4 deploy | Brief flurry of `auth_fail` from clients still racing the cookie hand-off (≤ 1× baseline). Self-resolves in seconds. |
| 5 min – 7 d | `auth_fail` rate **at or below baseline**. Dual-key verify silently catches old tokens. |
| Post-Step 6 | Brief flurry from any client whose 30-min access token straddled the cutover and was still cached. ≤ 1 minute. |

**Unhealthy rotation looks like:**

- **Sustained `auth_fail` spike** across the 7-day grace window → the dual-key code is broken; `JWT_SECRET_KEY_PREV` is empty, mistyped, or the decode block isn't actually reached. **Roll back immediately** (see below). The spike is observable real-time via `curl https://canslim.duckdns.org/api/admin/audit-log?limit=100 | jq '[.events[] | select(.event_type=="auth_fail")] | length'`.
- **`invalid_token` spikes from a single source_ip** → likely an attacker holding the old leaked key trying to forge tokens. The new `SECRET_KEY` correctly rejects them. Confirm by IP grep against your known-operator list, then add a rate-limit or block at Caddy.

---

## Rollback

If anything in Steps 4–5 looks wrong:

1. SSH to VPS, edit `.env` to swap back: `JWT_SECRET_KEY=<old>` (the value still in `JWT_SECRET_KEY_PREV` for one more minute, or recoverable from your password-manager backup).
2. `docker-compose restart backend` (no rebuild needed — env reload only). The container picks up the swapped values.
3. The dual-key code path harmlessly degrades to single-key (it just falls through the empty-`PREV` branch).
4. No data loss. Every active session is restored to the old secret in under 30 seconds.

If you cannot recover the old secret (password-manager wipe + already past Step 6) → force-logout all users by flipping `is_active` and let them re-authenticate via Google. Inconvenient but not destructive.

---

## What rotation does NOT cover

- **Google `client_id`** — this is a public identifier (frontend includes it in the OAuth handshake), not a secret. Rotation is a Google Cloud Console operation, not an `.env` change, and follows a separate playbook (TODO when first needed).
- **`FMP_API_KEY`** — rotated through the FMP dashboard. Not session-bearing, so no grace period needed. Just edit `.env` and `docker-compose restart`.
- **VPS SSH keys** — handled in `~/.ssh/authorized_keys` on the VPS. Out of scope for app-level rotation.

---

## Tabletop drill

Once a year (recommended: same week as the annual rotation), run a no-op drill:

1. Generate a new secret but write it to `JWT_SECRET_KEY_PREV` instead of `JWT_SECRET_KEY` (i.e. swap the convention so the *current* key stays primary).
2. Deploy the dual-key patch.
3. Hit `/api/auth/refresh` from a known-good session → must succeed.
4. Watch the audit log for one hour → `auth_fail` rate must stay flat.
5. Roll back.

This validates that the dual-key code path actually works when you need it, instead of discovering during a real incident that step 3 has been broken for 8 months.
