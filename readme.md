## Roadmap

(done) Fix voice (0 only) + dédoublonnage

Export JSONL par morceau (events triés, accords triés)

Baseline DP/Viterbi + metrics

Définir score jouabilité (même simple)

Ensuite seulement : modèle ML/NN + reranking jouabilité

## Convert gp5 to midi

```bash
python gp_to_midi.py --input ../sf.gp5 --output ../sf.mid
```

## Launch web server

### Dev

Frontend :
```bash
.\frontend\run -d
```

Backend :
```bash
cd python
python server.py --dev
```

### Prod

Frontend :
```bash
.\frontend\run -b
```

```bash
docker build -t tabify .
docker run -p 8000:8000 \
  -e GOOGLE_CLIENT_ID=xxx \
  -e GOOGLE_CLIENT_SECRET=yyy \
  -e SECRET_KEY=long-random-string \
  -e APP_BASE_URL=https://yourdomain.com \
  tabify
```

---

## TODO — Authentication layer (next session)

### Google Cloud Console (manual)
- Create a project → APIs & Services → Credentials → OAuth 2.0 Client ID
- Application type: **Web application**
- Add authorized redirect URI: `http://localhost:8000/api/auth/callback` (+ prod domain)
- Copy **Client ID** and **Client Secret** into env vars

### Frontend code changes (not yet done)
- `frontend/src/classes/network/Api.ts` — remove the localStorage short-circuit in `checkAuth()` so it always calls `/api/me` (currently skips the server call if no token in localStorage, which blocks cookie-based auth)
- `frontend/src/classes/Client.ts` — fix routing: `onNotConnected()` → `LoginPage`, `onConnected()` → `TabifyPage`
- `frontend/src/classes/pages/login/popups/LoginPopup.ts` — replace the email/password form with a single "Login with Google" button that redirects to `GET /api/auth/google`

### Backend already done
- `GET /api/auth/google` — redirects to Google OAuth
- `GET /api/auth/callback` — exchanges code, sets signed httponly session cookie, redirects to `/`
- `POST /api/auth/logout` — clears cookie
- `GET /api/me` — validates cookie, returns user `{email, name, picture}`
- Auth guard on `/api/tabify` and `/api/suggest-params`
- `ALLOWED_EMAILS` env var for access control (empty = allow all Google accounts)

### Tune profile (run once after setup)
```bash
cd python
python tune_viterbi.py --phase profile \
  --dataset_dir ../dev_folder/jsonl_dataset \
  --n_profile_files 2000 \
  --profile_out tune_profile.json
```

---

## TODO — Paywall (after auth)

Model: **freemium** — N free conversions/month, then paid subscription via Stripe.

### Implementation order
1. Finish auth (Google OAuth) first
2. Add **Supabase** (managed Postgres) — create project at supabase.com, copy connection string into `DATABASE_URL` env var. Minimum schema: `users(email, plan, conversions_used, stripe_customer_id)`
3. Stripe integration

### Backend to build
- `POST /api/checkout` — create Stripe Checkout session → return redirect URL
- `POST /api/stripe/webhook` — handle payment success / cancellation / renewal
- Paywall middleware on `/api/tabify` and `/api/suggest-params` (check plan + conversions_used)

### Frontend to build
- Usage counter display (e.g. "3/5 conversions used this month")
- Upgrade prompt when limit is hit
- "Upgrade" button → calls `/api/checkout` → redirects to Stripe
