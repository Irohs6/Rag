# Feuille de route — RAG against the machine (v2.0)

## 1. Suivre une vraie question dans le corpus

- [x] Placer vLLM dans `data/raw/vllm-0.10.1/`, ranger les quatre datasets dans `data/datasets/` et lancer l'aide de la moulinette sur le poste.
- [ ] Prendre une question dans `UnansweredQuestions/dataset_docs_public.json`. Retrouver son `question_id` dans le fichier du même nom sous `AnsweredQuestions/`.
- [ ] Ouvrir le fichier indiqué par la première source du corrigé et lire le passage aux positions indiquées. Quels mots de ce passage permettent de répondre à la question ?
- [ ] Refaire l'exercice avec `dataset_code_public.json`. Repérer la fonction ou la classe concernée.

Le corrigé sert à vérifier la recherche. Lors d'une recherche réelle, le programme reçoit la question et doit retrouver lui-même le fichier et la plage de caractères.

## 2. Préparer le projet

- [x] Initialiser le dépôt, son arborescence et sa licence.
- [ ] Compléter la première ligne du README avec le login 42.
- [x] Configurer Python 3.10+, uv, Pydantic, Python Fire, tqdm, flake8 et mypy. Ajouter `pyproject.toml` et `uv.lock`, puis vérifier `uv sync`.
- [x] Ajouter les règles `install`, `run`, `debug`, `clean` et `lint` au Makefile. Pour `lint`, exécuter `flake8 .` puis `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`.

## 3. Lire les fichiers et les découper

- [ ] Implémenter les huit modèles Pydantic de la section VI.4 du [sujet français](subject_fr.md). Commencer par `MinimalSource`, qui décrit le chemin et les deux positions d'un passage.
- [ ] Découper d'abord un fichier Markdown, puis un fichier Python, avec une stratégie adaptée à chaque format.
- [ ] Tester un fichier plus long que 2 000 caractères. Pour chaque passage, vérifier que le chemin correspond au fichier d'origine, que ses indices permettent d'en relire le texte exact et que sa longueur respecte `--max_chunk_size` ainsi que le plafond de 2 000 caractères.

## 4. Retrouver les sources et mesurer le résultat

- [ ] Construire un index BM25 ou TF-IDF et le sauvegarder dans `data/processed/` avec la commande `index`.
- [ ] Lancer `search` sur la question de l'étape 1. Examiner les cinq premiers passages et comparer leur emplacement aux sources du corrigé.
- [ ] Implémenter `search_dataset` pour écrire un JSON conforme à `StudentSearchResults`. Séparer les dossiers de sortie selon le dataset.
- [ ] Implémenter `evaluate`. Pour chaque question, compter les sources de référence retrouvées parmi les k premiers passages, dans le même fichier, avec un seuil d'IoU de 0,05. Lancer séparément la moulinette fournie pour vérifier les résultats. Le programme ne doit jamais l'appeler.
- [ ] Mesurer le recall@5 sur les deux datasets. Seuils requis : 80 % sur la documentation et 50 % sur le code. Mesurer aussi l'indexation du corpus (5 minutes maximum) et la recherche sur 200 questions (90 secondes maximum).

## 5. Générer les réponses et préparer le rendu

- [ ] Construire le contexte avec la question et les passages retrouvés en respectant le budget de tokens de Qwen/Qwen3-0.6B. Avec `answer`, vérifier sur une question connue que la réponse s'appuie sur le passage.
- [ ] Implémenter `answer_dataset` à partir des résultats de recherche sauvegardés. Valider la sortie avec `StudentSearchResultsAndAnswer`.
- [ ] Vérifier les six commandes via `uv run python -m src`, les chemins configurables et les barres de progression. Essayer une question vide ou absurde, `k=0`, un fichier absent et un JSON invalide. Chaque cas doit être traité sans exception non gérée.
- [ ] Vérifier les annotations, docstrings et ressources ouvertes. Documenter en anglais dans le README les choix, exemples, difficultés, usages de l'IA, temps et scores mesurés, y compris l'effet de la taille des passages.

## Bonus après validation de la partie obligatoire

- [ ] Index vectoriel avec un modèle léger sur CPU.
- [ ] Recherche hybride combinant classements lexical et sémantique.
- [ ] Réindexation des seuls fichiers modifiés.
- [ ] Cache de l'index et des résultats de recherche.
- [ ] API HTTP locale.
