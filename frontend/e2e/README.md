# End-to-end suite

Cross-page number-consistency tests, added 2026-09-09 after three bugs
shipped where the same figure disagreed between the Command Center, the AI
Portfolio chart, and the Edge hero.

## Running

```bash
cd frontend
npm run e2e          # builds the frontend, then runs the suite
npm run e2e:ui       # interactive mode
```

`npm run e2e` starts its own backend (`scripts/e2e_server.py`) against a
throwaway SQLite database and serves the built frontend from it — one uvicorn
process provides both API and UI. **It never touches the dev or production
database.** To run against an already-running instance instead:

```bash
E2E_BASE_URL=http://127.0.0.1:8001 npx playwright test
```

## The fixture is the point

`scripts/e2e_server.py` seeds a book where the **live value (10500) differs
from the newest snapshot (11000)**, and writes **no snapshot for today**.

That is the pre-market condition under which the September bugs were visible.
Every pre-existing unit test seeded those two EQUAL, so both ends of every
window matched by construction and the bugs were invisible. If you change the
fixture, keep them different — `the fixture itself` spec fails if you don't.

The scheduler is disabled and the DB is static, so nothing moves between page
loads. That is deliberate: it removes live-market drift as a source of
flakiness rather than trying to tolerate it.

## Local browser launch (no sudo)

`npx playwright install --with-deps chromium` needs root. CI has it; a dev
box may not. If the browser fails with `error while loading shared
libraries: libnspr4.so`, point the loader at the unpacked deb libs:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/pw-libs/root/usr/lib/x86_64-linux-gnu:$HOME/.local/lib/pw-libs/root/usr/lib"
npx playwright test
```

## Adding a surface

If you add a page or endpoint that reports a portfolio value or a windowed
return, add it to the `API surfaces agree` spec. The backend equivalent lives
in `tests/test_portfolio_surface_consistency.py::_collect_surfaces`.
