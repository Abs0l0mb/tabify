## Roadmap

(done) Fix voice (0 only) + dédoublonnage

Export JSONL par morceau (events triés, accords triés)

Baseline DP/Viterbi + metrics

Définir score jouabilité (même simple)

Ensuite seulement : modèle ML/NN + reranking jouabilité

## Launch web server

### Dev

Frontend :
```bash
.\frontend\run -d
```

Backend :
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Prod

Frontend :
```bash
.\frontend\run -b
```

```bash
# Build frontend first: cd frontend && ./run -p
docker build -t tabify .
docker run -p 8000:8000 tabify
```
