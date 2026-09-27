# Hosting guide — Streamlit Community Cloud + Supabase

```
Phone / laptop browser ──HTTPS──▶ Streamlit Community Cloud (app.py, free)
                                        │  DATABASE_URL (TLS)
                                        ▼
                                 Supabase Postgres (free)
                                   ├─ test_logs  (records + evidence PNG, hash-chained)
                                   └─ users      (officer / admin accounts, scrypt hashes)
```

Without `DATABASE_URL` the app falls back to a local SQLite file in `data/` (for offline/dev use).

## 1. Create the database (Supabase, ~3 min)

1. Sign up at <https://supabase.com> → **New project**. Region: **Mumbai (ap-south-1)**. Save the database password.
2. Project → **Connect** → **Session pooler** → copy the URI. It looks like
   `postgresql://postgres.<ref>:<password>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres`
   Append `?sslmode=require`. (Use the *pooler* URI: the "direct" one is IPv6-only and fails from Streamlit Cloud.)
3. Nothing else — tables and indexes are created automatically on first start, with Row-Level Security
   enabled so Supabase's public REST API cannot read them.

## 2. Generate the device signing key (once, keep forever)

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

To carry over records sealed locally, instead run the migration (step 5) — it prints the local key to reuse.

## 3. Deploy the app (Streamlit Community Cloud)

1. Sign in at <https://share.streamlit.io> with GitHub → **Create app** → pick this repository, branch `main`,
   main file `app.py`.
2. **Advanced settings** → Python version **3.11** → paste secrets (template: `.streamlit/secrets.toml.example`):

   ```toml
   DATABASE_URL    = "postgresql://postgres.<ref>:<password>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"
   DEVICE_HMAC_KEY = "<64 hex chars from step 2>"
   ADMIN_USERNAME  = "admin"
   ADMIN_PASSWORD  = "<strong password>"
   ```
3. **Deploy**. Sign in as the admin, open **Administration**, create officer accounts.
   After the first sign-in you may delete `ADMIN_PASSWORD` from secrets (it is only used when no users exist).

The app URL is HTTPS, so the phone camera works.

## 4. Managing the database

| Task | Where |
|---|---|
| Create / deactivate officers, reset passwords | App → **Administration** (admin only) |
| Full backup (records + evidence images + CSV + integrity report) | App → Administration → **Prepare full evidence archive (ZIP)** |
| Check nothing was altered | App → Evidence Ledger → **Verify entire chain** |
| Browse raw tables / run SQL | Supabase dashboard → **Table Editor** / **SQL Editor** |
| Point-in-time backups | Supabase → Database → Backups (daily on paid plans; download the ZIP regularly on free) |

Rules for evidence integrity:
* **Never edit or delete rows in `test_logs`.** Any change breaks that record's seal and every later chain link
  (this is by design — verification will flag it).
* **Never change `DEVICE_HMAC_KEY`** once records exist.
* Free Supabase projects pause after 7 days without activity — open the app (or dashboard) weekly, or upgrade.
* Free tier = 500 MB database ≈ 3,000 records (each stores a ~160 KB calibrated image).

Useful SQL (Supabase → SQL Editor):

```sql
SELECT result, COUNT(*) FROM test_logs GROUP BY result;                        -- summary
SELECT id, timestamp, operator_id, result FROM test_logs ORDER BY id DESC LIMIT 20;
SELECT username, role, active FROM users;                                      -- accounts
SELECT pg_size_pretty(pg_database_size(current_database()));                   -- space used
```

## 5. (Optional) Copy existing local records to Supabase

```bash
source .venv/bin/activate
python scripts/migrate_to_postgres.py "postgresql://...pooler.supabase.com:5432/postgres?sslmode=require"
```

The target must be empty. The script preserves record ids, re-verifies every seal in Postgres and prints the
`DEVICE_HMAC_KEY` value to use in Streamlit secrets.

## Local development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py          # SQLite in ./data, first visit shows "create administrator"
pytest                        # runs every test on SQLite and on an embedded Postgres
```
