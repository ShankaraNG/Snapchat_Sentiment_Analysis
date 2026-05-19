import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline as skpipe
from sklearn.preprocessing import FunctionTransformer
from ml_build.services import preprocessing as preprocessing
from ml_build.logger import get_logger

log = get_logger('LDA PipelineBuilder')

def ldapipelinebuilder(maxdiff, mindiff, k, method):
    try:
        log.info("Building LDA topic modeling pipeline")
        lda_pipeline = skpipe(steps=[
            ('lda_preprocessing', FunctionTransformer(preprocessing.lda_preprocessing)),
            ('countvectorizer', CountVectorizer(max_df=maxdiff, min_df=mindiff, ngram_range=(1, 2))),
            ('lda_model', LatentDirichletAllocation(n_components=k, random_state=42, learning_method=method))
        ])
        log.info("LDA pipeline built successfully")
        return lda_pipeline
    except Exception as e:
        log.error(f"Error building LDA pipeline: {e}")
        return None