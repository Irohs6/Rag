# RAG against the machine

*Répondrez-vous à mes questions ?*  
**Version du sujet : 2.0**

> Traduction et adaptation Markdown du sujet anglais fourni. Le PDF original fait foi en cas de différence de mise en forme ou de transcription. Les noms des modèles, champs, commandes et chemins restent en anglais pour correspondre à l'interface évaluée.

## Sommaire

1. [Préambule](#i-préambule)
2. [Avant-propos](#ii-avant-propos)
3. [Consignes concernant l'IA](#iii-consignes-concernant-lia)
4. [Introduction](#iv-introduction)
5. [Consignes communes](#v-consignes-communes)
6. [Partie obligatoire](#vi-partie-obligatoire)
7. [Évaluation](#vii-évaluation)
8. [Exigences du README](#viii-exigences-du-readme)
9. [Bonus](#ix-bonus)
10. [Rendu et évaluation par les pairs](#x-rendu-et-évaluation-par-les-pairs)

## I. Préambule

Vous allez construire une chaîne de traitement qui lit et indexe un dépôt de code, retrouve les passages les plus pertinents pour une question, les transmet à un petit modèle local pour générer une réponse fondée sur ces sources, puis mesure la qualité de la recherche avec le recall@k.

## II. Avant-propos

Les connaissances d'un modèle de langage sont figées après son entraînement. Une bibliothèque récente, un dépôt privé ou un événement récent peuvent lui être inconnus, et le réentraîner après chaque changement coûterait trop cher. La génération augmentée par recherche (RAG) lui apporte des informations externes au moment de répondre. Dans ce projet, cette source est un grand dépôt de code. La difficulté consiste à retrouver les quelques passages utiles sans manquer la réponse ni noyer le petit modèle sous du contenu inutile.

## III. Consignes concernant l'IA

L'IA peut aider à écrire du code ou de la documentation et à explorer des idées et des explications. Ses réponses peuvent cependant être fausses ou biaisées. Comprenez, vérifiez et assumez ce que vous utilisez, et faites relire votre travail par d'autres étudiants.

Les objectifs pédagogiques sont d'automatiser les tâches répétitives, d'améliorer la formulation des demandes, de comprendre les risques de l'IA et de développer la résolution de problèmes et la collaboration. Réfléchissez avant de formuler une demande, utilisez un vocabulaire précis et prenez l'habitude de tester et de remettre en question les résultats produits.

Vous pouvez demander à l'IA des idées de tests ou une proposition de conception de parseur, puis vérifier et améliorer ce travail avec un pair. Copier une fonction complète que vous ne savez pas expliquer pendant l'évaluation est un mauvais usage.

## IV. Introduction

Après le *function calling* de `call_me_maybe`, ce projet aborde le RAG. Au lieu de réentraîner le modèle pour lui apporter des connaissances, vous lui donnez accès à une source externe au moment de répondre. Le RAG comporte quatre étapes :

1. **Indexation :** organiser les données pour permettre la recherche.
2. **Recherche :** comparer la question à l'index et sélectionner les passages pertinents.
3. **Augmentation du contexte :** filtrer ces passages et les placer dans la fenêtre de contexte du modèle.
4. **Génération :** produire une réponse à partir de ce contexte.

Lisez tout le sujet avant de coder : chaque étape fournit les données nécessaires à la suivante.

## V. Consignes communes

### V.1 Règles générales

- Utilisez Python 3.10 ou une version ultérieure et respectez le standard `flake8`.
- Gérez les exceptions proprement. Un plantage non géré pendant l'évaluation rend le programme non fonctionnel. Gérez les ressources sans fuite et privilégiez les gestionnaires de contexte.
- Annotez les paramètres, les valeurs de retour et, lorsque c'est pertinent, les variables. Toutes les fonctions doivent passer `mypy` sans erreur.
- Documentez fonctions et classes avec des docstrings conformes à PEP 257, en précisant leur rôle, leurs paramètres et leurs valeurs de retour quand cela s'applique.

### V.2 Makefile

Les règles obligatoires sont :

| Règle | Action |
| --- | --- |
| `install` | Installer les dépendances avec le gestionnaire de paquets choisi. |
| `run` | Lancer le programme principal. |
| `debug` | Lancer le programme sous un débogueur tel que `pdb`. |
| `clean` | Supprimer les fichiers temporaires et les caches. |
| `lint` | Lancer `flake8 .` et la commande `mypy` imposée. |

La commande `mypy` demandée est :

```sh
mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

Une règle `lint-strict` facultative peut lancer `flake8 .` et `mypy . --strict`.

### V.3 Conseils supplémentaires

- Écrivez des tests, par exemple avec `pytest` ou `unittest`, en couvrant les cas limites. Les programmes de test ne sont pas rendus ni notés.
- Ajoutez un `.gitignore` pour les fichiers générés par Python.
- L'usage d'un environnement virtuel pendant le développement est recommandé.

### V.4 Exigences propres au projet

- Utilisez **Pydantic** pour valider les modèles de données échangés entre les étapes. Les classes de service et d'orchestration, comme l'indexeur, le moteur de recherche ou le pipeline, n'ont pas besoin d'être des modèles Pydantic.
- `flake8` s'applique également aux fichiers des bonus.
- Utilisez **uv** comme gestionnaire de projet et de paquets. L'évaluateur et la moulinette exécutent `uv sync`.
- Exposez une interface en ligne de commande avec **Python Fire** et affichez des barres de progression **tqdm** pour les opérations longues.
- Le modèle par défaut est **Qwen/Qwen3-0.6B**. D'autres modèles sont permis si le projet fonctionne toujours avec Qwen/Qwen3-0.6B.
- En dehors de ces outils imposés, vous pouvez utiliser les bibliothèques de votre choix.

## VI. Partie obligatoire

Créez un système RAG qui répond à des questions sur le dépôt **vLLM fourni**. Vous devez indexer ses fichiers, retrouver les passages pertinents, générer des réponses avec Qwen/Qwen3-0.6B et évaluer la recherche avec le recall@k. La correction porte sur la pertinence des emplacements retrouvés et sur des réponses fondées sur ces sources.

### VI.1 Indexation du dépôt

Lisez les fichiers que vous jugez utiles dans le dépôt vLLM fourni, découpez-les en passages (*chunks*) et enregistrez un index dans `data/processed/`. L'indexation de tout le corpus doit prendre **5 minutes au maximum**.

Implémentez **deux méthodes de découpage distinctes** : l'une pour le code Python, l'autre pour Markdown et le texte. Pour la recherche, implémentez au moins une méthode **lexicale classique : TF-IDF ou BM25**. Vous pouvez ajouter d'autres méthodes en complément.

L'option `--max_chunk_size` règle la taille des passages, avec **2 000 caractères** par défaut. Ne renvoyez jamais un passage de plus de 2 000 caractères : le `max_context_length` de la moulinette vaut 2 000 et un seul passage trop long invalide toute la sortie. Des passages plus petits sont autorisés ; présentez leur effet sur le recall@k.

Les choix faits pendant l'indexation influencent la recherche : une question peut reformuler une idée ou citer exactement un identifiant du code.

### VI.2 Recherche

Pour une question, renvoyez les `k` emplacements de source les mieux classés. Chaque résultat contient `file_path`, `first_character_index` et `last_character_index`. Le passage couvert ne doit pas dépasser 2 000 caractères.

Le chemin doit correspondre **exactement** au chemin du corpus, préfixe inclus, par exemple `data/raw/vllm-0.10.1/docs/features/lora.md`. La moulinette compare les chemins caractère par caractère.

La recherche doit fonctionner sur une question isolée et sur un lot de questions provenant d'un dataset JSON. Sur les jeux de référence, il faut obtenir **au moins 80 % de recall@5 pour la documentation** et **50 % pour le code**.

### VI.3 Génération des réponses

Transmettez les passages retrouvés à Qwen/Qwen3-0.6B en respectant son budget de tokens. Produisez une réponse en langage naturel et une sortie JSON structurée conforme aux modèles Pydantic fournis.

Les réponses doivent être compréhensibles, répondre à la question et rester fondées sur les sources, sans hallucination majeure. Le sujet tient compte des limites de raisonnement de ce petit modèle : une formulation parfaite n'est pas une condition éliminatoire stricte. La correction privilégie la qualité de la recherche, l'ancrage dans les sources et la stratégie de prompt. Une réponse cohérente et principalement fondée sur les documents peut être acceptée même si elle reste partiellement incomplète.

### VI.4 Modèles de données

Implémentez les modèles Pydantic ci-dessous. Les noms de champs et leurs types constituent la structure minimale demandée :

```python
import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
```

Vous pouvez ajouter des modèles et des champs si nécessaire, à condition de respecter cette structure minimale.

### VI.5 Sorties

La sortie de recherche suit le modèle `StudentSearchResults` ; la sortie des réponses suit `StudentSearchResultsAndAnswer`. Le `file_path` est relatif à la racine du projet et reprend exactement le chemin du fichier indexé. Les indices indiquent le début et la fin du passage dans ce fichier.

Exemple de résultat de recherche :

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How to configure OpenAI server?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100
        },
        {
          "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
          "first_character_index": 267,
          "last_character_index": 400
        }
      ]
    }
  ],
  "k": 10
}
```

Le format des réponses reprend ce résultat et ajoute un champ `answer` de type chaîne à chaque élément de `search_results`, par exemple `"answer": "To configure the OpenAI compatible server in vLLM..."`.

### VI.6 Interface en ligne de commande

Utilisez Python Fire. Chaque commande doit être appelée sous la forme `uv run python -m src <commande> [options]`. Les commandes et options minimales sont :

| Commande | Fonction |
| --- | --- |
| `index --max_chunk_size <int>` | Lire `data/raw/` et créer l'index dans `data/processed/`. |
| `search <query> --k <int>` | Rechercher les `k` meilleures sources pour une question. |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Rechercher pour toutes les questions d'un dataset JSON et enregistrer `StudentSearchResults`. |
| `answer <query> --k <int>` | Répondre à une question à partir des passages retrouvés. |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Produire des réponses à partir des résultats sauvegardés et enregistrer `StudentSearchResultsAndAnswer`. |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Mesurer le recall@k à partir des sources de référence pour vos propres essais. |

Tous les chemins d'entrée et de sortie doivent être configurables par des arguments CLI et ne doivent pas être codés en dur. Affichez `tqdm` pendant les longues opérations. Traitez proprement les requêtes vides ou absurdes, `k=0`, les fichiers absents et les JSON mal formés, sans traceback non géré.

La commande `evaluate` sert à vos itérations. Le recall@k officiel est calculé par l'exécutable de la moulinette fourni. Votre application ne doit **jamais importer ou appeler la moulinette**.

### VI.7 Déroulement complet

Pendant la soutenance, une chaîne automatisée exécute `index` → `search_dataset` → `evaluate_student_search_results` de la moulinette → `answer_dataset`. Les commandes pour une seule question restent disponibles pour déboguer.

L'arborescence attendue pendant l'évaluation est :

```text
src/
pyproject.toml
uv.lock
README.md
data/
  raw/                                  # dépôt vLLM fourni
  processed/                            # index généré
  datasets/
    UnansweredQuestions/
    AnsweredQuestions/
  output/
    search_results/<DatasetScope>/
    search_results_and_answer/<DatasetScope>/
```

Les scripts de référence fournissent leurs propres chemins d'entrée et de sortie. Si l'interface ou l'arborescence n'est pas respectée, les vérifications correspondantes échouent parce que les scripts ne peuvent pas s'exécuter.

Exemple de commandes (sur une seule ligne dans le terminal, ou avec la continuation de ligne du shell) :

```sh
uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Pour cet exemple d'évaluation, renommez d'abord `moulinette-ubuntu` ou `moulinette-fedora` en `moulinette`. Le premier argument positionnel est votre résultat de recherche, le second est le dataset corrigé `AnsweredQuestions`. Les scores illustrés dans le PDF montrent seulement le format de la sortie : **ce ne sont pas les seuils requis**.

Séparez les dossiers `--save_directory` pour `UnansweredQuestions` et `AnsweredQuestions`. Les datasets publics peuvent avoir le même nom de fichier : un dossier commun écraserait les sorties précédentes.

## VII. Évaluation

### VII.1 Calcul du recall@k

Pour chaque question, le recall@k est la proportion de sources correctes retrouvées dans vos `k` premiers résultats. Une source est trouvée si le passage renvoyé appartient au **même fichier** et chevauche la plage de caractères de référence. Le seuil d'intersection sur union (IoU) est **0,05**. Il n'est pas nécessaire de reproduire exactement les limites du passage, mais il faut retrouver la bonne zone du bon fichier.

La moulinette valide votre sortie et affiche le recall pour plusieurs valeurs de `k`.

### VII.2 Seuils de performance

| Mesure | Exigence |
| --- | ---: |
| Indexation de tout le corpus | ≤ 5 minutes |
| Recherche pour 200 questions | ≤ 90 secondes |
| Recall@5 sur la documentation | ≥ 80 % |
| Recall@5 sur le code | ≥ 50 % |

## VIII. Exigences du README

Le fichier `README.md` à la racine doit être **en anglais**. Sa **première ligne, en italique**, doit reprendre exactement la formule demandée dans le sujet, avec les vrais logins :

> *This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].*

Il doit contenir des sections Description, Instructions et Resources. Cette dernière liste les références classiques et précise comment l'IA a été utilisée, pour quelles tâches et dans quelles parties du projet. Le README doit aussi expliquer :

- L'architecture du système et les échanges entre ses composants.
- La stratégie de découpage des fichiers.
- L'algorithme de recherche et le classement.
- Les scores de recall@k et les performances.
- Les choix de conception et les difficultés rencontrées.
- Des exemples de commandes.

## IX. Bonus

Les bonus ne sont évalués **qu'après validation de toute la partie obligatoire**, y compris sa robustesse face à des cas inhabituels proposés par l'évaluateur. Chacun des cinq bonus fonctionnels vaut un point. Ils doivent tourner sur une **machine du campus sans GPU** ; une simple mention dans le README ne suffit pas.

1. **Embeddings sémantiques :** ajouter un index vectoriel à côté de l'index lexical avec un petit modèle CPU tel que `all-MiniLM-L6-v2`.
2. **Recherche hybride :** fusionner les classements lexicaux et sémantiques.
3. **Indexation incrémentale :** ne réindexer que le fichier modifié.
4. **Cache :** mettre en cache l'index et les résultats de recherche pour accélérer le démarrage et les requêtes répétées.
5. **API HTTP locale :** permettre à un autre client d'effectuer une recherche et d'obtenir une réponse sans passer par la CLI.

## X. Rendu et évaluation par les pairs

Rendez votre dépôt Git. Seul son contenu sera examiné. Il doit contenir :

- `src/` avec votre implémentation.
- `pyproject.toml` et `uv.lock`.
- Un `Makefile` avec `install`, `run`, `debug`, `clean` et `lint`.
- Un `README.md` complet et les éventuels fichiers de configuration nécessaires.

**Ne versionnez pas les gros fichiers de données, les poids du modèle ou les sorties générées.** L'évaluateur les fournira ou les générera. La pile d'apprentissage profond et les poids peuvent occuper plusieurs gigaoctets : prévoyez suffisamment d'espace disque.

Pendant l'évaluation, il peut vous être demandé de modifier rapidement quelques lignes de code, un comportement, un affichage ou une structure de données. Vous devez pouvoir expliquer et adapter votre propre travail.
