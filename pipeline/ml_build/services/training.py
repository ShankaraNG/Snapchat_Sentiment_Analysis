import random

import matplotlib.pyplot as plt
import pandas as pd
from ml_build.services import preprocessing as preprocessing
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from ml_build.config_loader import load_config
from ml_build.logger import get_logger
import joblib
import os

log = get_logger('Training')

BASE_DIR = os.path.dirname(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))

def lda_model_training(pipeline, data_df, artifacts_path, model_name, model_path):
    try:
        log.info("Starting LDA model training") 
        tokenized_texts = [str(text).split() for text in data_df if pd.notna(text)]
        dictionary = Dictionary(tokenized_texts)
        log.info("Fitting LDA pipeline to the data")
        pipeline.fit(data_df)
        log.info("LDA pipeline fitted successfully, calculating perplexity and coherence scores")
        results = pipeline.transform(data_df)
        vectorizer = pipeline.named_steps['countvectorizer']
        words = vectorizer.get_feature_names_out()
        preprocessed_text = pipeline.named_steps['lda_preprocessing'].transform(data_df)
        vectorizer_matrix = vectorizer.transform(preprocessed_text)
        perplexity = pipeline.named_steps['lda_model'].perplexity(vectorizer_matrix)
        log.info(f"LDA Model Perplexity: {perplexity}")
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "lda_perplexity_score.txt")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"LDA Model Perplexity: {perplexity}\n")
        log.info("saving LDA model perplexity score to file")
        log.info("Calculating LDA model coherence score")
        topics = []
        lda_model = pipeline.named_steps['lda_model']
        sample_size = min(5000, len(tokenized_texts))
        sampled_texts = random.sample(tokenized_texts, sample_size)
        for topic in lda_model.components_:
            topic_words = [words[idx] for idx in topic.argsort()[-5:]]
            topics.append(topic_words)
        coherence_model = CoherenceModel(topics=topics, texts=sampled_texts, dictionary=dictionary, coherence='c_v', processes=1)
        coherence_score = coherence_model.get_coherence()
        log.info(f"LDA Model Coherence Score: {coherence_score}")
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "lda_coherence_score.txt")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"LDA Model Coherence Score: {coherence_score}\n")
        log.info("saved LDA model coherence score to file")
        n_top_words = 20
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "topics.txt")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("")
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [words[i] for i in top_indices]
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(f"TOPIC {topic_idx + 1}: ")
                f.write(", ".join(top_words) + "\n")
        log.info("Saved LDA topics to file")
        full_model_path = os.path.join(BASE_DIR, model_path, f"{model_name}.joblib")
        if not os.path.exists(os.path.join(BASE_DIR, model_path)):
            os.makedirs(os.path.join(BASE_DIR, model_path))
        joblib.dump(pipeline, full_model_path)
        log.info("LDA model saved successfully")
        log.info("LDA model training completed successfully")
        return 'successfull'
    except Exception as e:
        log.error(f"Failed to train LDA model: {str(e)}")
        return None


def sentiment_pipeline_saving(pipeline, data_df, artifacts_path, model_name, model_path):
    try:
        log.info("Starting sentiment pipeline training")
        data_df['predicted_sentiment'] = pipeline.fit_transform(data_df['translated_text'])
        log.info("Sentiment pipeline fitted successfully, calculating accuracy score and classification report")
        accuracyscore = accuracy_score(data_df['score_sentiment'], data_df['predicted_sentiment'])
        log.info(f"Sentiment Model Accuracy Score: {accuracyscore}")
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "sentiment_accuracy_score.txt")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Sentiment Model Accuracy Score: {accuracyscore}\n")
        log.info("Saved sentiment accuracy score to file")
        clf_report = classification_report(data_df['score_sentiment'], data_df['predicted_sentiment'], target_names=['Negative', 'Neutral', 'Positive'])
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "sentiment_classification_report.txt")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Sentiment Model Classification Report:\n{clf_report}\n")
        log.info("Saved sentiment classification report to file")
        ConfusionMatrixDisplay.from_predictions(
            data_df['score_sentiment'], 
            data_df['predicted_sentiment'], 
            display_labels=['Negative', 'Neutral', 'Positive'],
            cmap='Blues'
        )
        plt.title('Roberta vs. Actual Snapchat Ratings')
        output_image_path = os.path.join(BASE_DIR, artifacts_path, "sentiment_confusion_matrix.png")
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        log.info("Saved sentiment confusion matrix to file")
        full_model_path = os.path.join(BASE_DIR, model_path, f"{model_name}.joblib")
        if not os.path.exists(os.path.join(BASE_DIR, model_path)):
            os.makedirs(os.path.join(BASE_DIR, model_path))
        joblib.dump(pipeline, full_model_path)
        log.info("Sentiment model saved successfully")
        log.info("saving the sentiment data with predictions to file")
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "sentiment_predictions.csv")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        data_df.to_csv(output_file_path, index=False)
        log.info("Sentiment data with predictions saved successfully")
        log.info("Sentiment model training completed successfully")
        return 'successfull'
    except Exception as e:
        log.error(f"Failed to do the sentiment analysis: {str(e)}")
        return None