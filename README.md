# Projet TAL — Moteur de recherche sur CISI

Système de recherche d'information sur le corpus CISI, avec trois approches de scoring :

- `projet_nlp.py` — TF-IDF + similarité cosinus
- `projet_nlp_bm25.py` — BM25 (`rank_bm25`)
- `projet_nlp_all-mini-lm_model.py` — embeddings BERT (`all-MiniLM-L6-v2`), combinés avec TF-IDF

Chaque script tokenise les documents/requêtes avec spaCy (`en_core_web_sm`), calcule les scores de pertinence, et exporte les résultats au format `.REL`.

## Données

Le corpus se trouve dans `CISI/DATA/` :
- `CISI.ALLnettoye` — documents
- `CISI_dev.QRY` — requêtes
- `CISI_dev.REL` — gold standard (jugements de pertinence)

## Utilisation

```bash
python projet_nlp.py                  # génère CISI_results.REL
python projet_nlp_bm25.py             # génère CISI_results_bm25.REL
python projet_nlp_all-mini-lm_model.py  # génère CISI_results_BERT.REL
```

## Évaluation

```bash
python eval.py CISI/DATA/CISI_dev.REL CISI_results.REL
```

Affiche précision, rappel, F1, P@1 et P@5 par requête et en global.

## Dépendances

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
