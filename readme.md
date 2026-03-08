## Roadmap

(done) Fix voice (0 only) + dédoublonnage

Export JSONL par morceau (events triés, accords triés)

Baseline DP/Viterbi + metrics

Définir score jouabilité (même simple)

Ensuite seulement : modèle ML/NN + reranking jouabilité

## Launch web server

### Dev

```bash
python app.py
```

### Prod

```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:5000
```
