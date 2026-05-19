# Snapchat Review Sentiment Analysis & Topic Modelling

A machine learning pipeline that analyses Snapchat app reviews to predict sentiment (Positive, Negative, Neutral) and classify reviews into topics using LDA (Latent Dirichlet Allocation) and a pretrained RoBERTa model.

---

## Deployment

The Pipeline has been deployed in the Hugging face Platform with the below mentioned end point

https://shankarang-sentiment-topic-model.hf.space/ => Root End Point

https://shankarang-sentiment-topic-model.hf.space/health => Health End point

https://shankarang-sentiment-topic-model.hf.space/predict => Post end point for predictions

---

## Project Structure

```
Snapchat Sentiment Analysis/
│
├── pipeline/                          # Main application and ML pipeline
│   ├── app/                           # FastAPI application
│   │   ├── routes/
│   │   │   └── prediction.py          # POST /predict endpoint
│   │   ├── services/
│   │   │   ├── ldamodel.py            # LDA inference service
│   │   │   ├── sentimentmodel.py      # Sentiment inference service
│   │   │   └── topicloading.py        # Topic label loader
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── main.py                    # FastAPI app entrypoint
│   │
│   ├── ml_build/                      # ML training pipeline
│   │   ├── services/
│   │   │   ├── pipelinebuilderlda.py
│   │   │   ├── pipelinebuildersentiment.py
│   │   │   ├── pipelinerunner.py
│   │   │   ├── preprocessing.py
│   │   │   ├── testing.py
│   │   │   └── training.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── main.py                    # ML pipeline entrypoint
│   │
│   ├── artifacts/                     # Model evaluation outputs
│   │   ├── lda_coherence_score.txt
│   │   ├── lda_perplexity_score.txt
│   │   ├── lda_testing_results.txt
│   │   ├── sentiment_accuracy_score.txt
│   │   ├── sentiment_classification_report.txt
│   │   ├── sentiment_confusion_matrix.png
│   │   ├── sentiment_predictions.csv
│   │   ├── sentiment_testing_results.txt
│   │   └── topics.txt
│   │
│   ├── config/
│   │   └── config.yaml                # Configuration file
│   ├── data/                          # Input data
│   ├── logs/                          # Application and training logs
│   ├── model/                         # Saved model files
│   │   ├── snapchat_sentiment_model.joblib
│   │   ├── snapchat_topic_model.joblib
│   │   └── topic_labels.json
│   │
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── poetry.lock
│   └── requirements.txt
│
├── sanpchatPipeline.ipynb             # Exploratory analysis notebook
└── texttranslator.ipynb               # Translation analysis notebook
```

---

## Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/docs/) >= 2.0.0
- Docker (optional, for containerised deployment)

---

## Environment Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Snapchat Sentiment Analysis/pipeline"
```

### 2. Install Poetry

```bash
pip install poetry
```

### 3. Install dependencies

```bash
poetry install
```

### 4. Configure the application

Edit `config/config.yaml` with your settings:

```yaml
dataPath:
  path: data
  file: your_data_file.csv

contentcolumn:
  column_name: your_text_column

comparisoncolumn:
  column_name: your_rating_column

language:
  multilingual: false        # Set to true if data contains non-English text

robertamodel:
  model: cardiffnlp/twitter-roberta-base-sentiment

model:
  save_path: model
  sentiment_name: snapchat_sentiment_model
  lda_name: snapchat_topic_model

artifacts:
  save_path: artifacts

topicmapping:
  filepath: model
  filename: topic_labels.json

ldamodel:
  numberoftopics: 19
  learningmethod: batch

countvectorizerparams:
  maxdf: 0.9
  mindf: 5

ldacustomstopwords: []
```

---

## Running the ML Training Pipeline

This step trains the sentiment and LDA topic models and saves them to the `model/` directory.

```bash
poetry run python -m ml_build.main
```

This will:
1. Load and preprocess the data
2. Train and evaluate the RoBERTa sentiment pipeline
3. Train and evaluate the LDA topic model
4. Save all artifacts to `artifacts/`
5. Save models to `model/`

Training logs are saved to `logs/training.log`.

---

## Running the API

Once the models are trained, start the FastAPI server:

```bash
poetry run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive API docs are available at `http://localhost:8000/docs`.

---

## API Usage

### POST `/predict`

Accepts a list of review texts and returns sentiment and topic predictions for each.

**Request:**
```json
{
    "texts": [
        {
            "id": "review_001",
            "text": "The app keeps crashing after the update"
        },
        {
            "id": "review_002",
            "text": "Love the new filters on this app"
        },
        {
            "id": "review_003",
            "text": "Cannot login to my account, keeps saying wrong password"
        }
    ]
}
```

The `id` field is optional — requests without IDs are also accepted.

---

## Docker Deployment

### Build the image

```bash
docker build -t snapchat-sentiment .
```

### Run the container

```bash
docker run -p 8000:8000 snapchat-sentiment
```

### With a HuggingFace token (recommended to avoid rate limits)

```bash
docker run -p 8000:8000 -e HF_TOKEN=your_token_here snapchat-sentiment
```

---

## Topic Labels

The LDA model identifies 19 topics from Snapchat reviews:

| Topic | Label |
|-------|-------|
| 1 | Login & Account Access Issues |
| 2 | General App Frustration |
| 3 | Update & Crash Issues |
| 4 | Video & Call Quality |
| 5 | Friend Requests & Privacy |
| 6 | Discover & Content Feed |
| 7 | Filters & Streaks |
| 8 | Screenshots & Camera Features |
| 9 | Bugs & App Rating |
| 10 | Camera Roll & Bitmoji |
| 11 | Connectivity & Phone Issues |
| 12 | App Crashes & Reinstall |
| 13 | App Not Opening |
| 14 | Ads & Storage |
| 15 | Fun & Social Features |
| 16 | Notifications & Camera Settings |
| 17 | Snaps & Messaging Issues |
| 18 | Staying Connected with Family & Friends |
| 19 | Camera Quality & Device Comparison |

---

## Notebooks

Located in the root `Snapchat Sentiment Analysis/` folder:

- **`sanpchatPipeline.ipynb`** — Exploratory data analysis and pipeline prototyping
- **`texttranslator.ipynb`** — Analysis and testing of the translation component for multilingual reviews

---

## Model Performance

After training, evaluation artifacts are saved to the `artifacts/` folder:

- `sentiment_accuracy_score.txt` — Overall accuracy of RoBERTa predictions vs actual ratings
- `sentiment_classification_report.txt` — Precision, recall, and F1 per sentiment class
- `sentiment_confusion_matrix.png` — Visual confusion matrix
- `lda_perplexity_score.txt` — LDA model perplexity score
- `lda_coherence_score.txt` — LDA model coherence score
- `topics.txt` — Top words per topic
