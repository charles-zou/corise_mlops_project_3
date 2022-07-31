import datetime
import time
from pathlib import Path

import joblib
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


APP_DIR = Path(__file__).resolve().parent

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768,
        },
        "classifier": {
            "serialized_model_path": APP_DIR / "../data/news_classifier.joblib"
        },
    },
    "service": {"log_destination": APP_DIR / "../data/logs.out"},
}


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    # estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    # transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        """
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer`
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`
        """
        sentence_transformer_model = SentenceTransformer(
            self.config["model"]["featurizer"]["sentence_transformer_model"]
        )
        model = joblib.load(self.config["model"]["classifier"]["serialized_model_path"])

        featurizer = TransformerFeaturizer(
            dim=self.config["model"]["featurizer"][
                "sentence_transformer_embedding_dim"
            ],
            sentence_transformer_model=sentence_transformer_model,
        )

        self.pipeline = Pipeline(
            [
                ("transformer_featurizer", featurizer),
                ("classifier", model),
            ]
        )

    def predict_proba(self, model_input: dict) -> dict:
        """
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction probability scores across all labels

        Output format:
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2
            ...
        }
        """
        # Making only 1 pred at a time
        probs = self.pipeline.predict_proba([model_input["description"]])[0, :]
        return dict(zip(self.pipeline.classes_, probs))

    def predict_label(self, model_input: dict) -> str:
        """
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """
        # Making only 1 pred at a time
        pred = self.pipeline.predict([model_input["description"]])[0]
        return pred


app = FastAPI()

clf = NewsCategoryClassifier(config=GLOBAL_CONFIG)


@app.on_event("startup")
def startup_event():
    """
    1. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass
        any relevant config parameters from `GLOBAL_CONFIG` that are needed by NewsCategoryClassifier
    2. Open an output file to write logs, at the destimation specififed by
        GLOBAL_CONFIG['service']['log_destination']

    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    logger.add(GLOBAL_CONFIG["service"]["log_destination"])
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    """
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened
        in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
        {
            'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
            'request': dictionary representation of the input request,
            'prediction': dictionary representation of the response,
            'latency': time it took to serve the request, in millisec
        }
    3. Construct an instance of `PredictResponse` and return
    """
    start_time = time.monotonic()
    request_dict = request.dict()
    response_dict = {
        "scores": clf.predict_proba(request_dict),
        "label": clf.predict_label(request_dict),
    }
    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        "request": request_dict,
        "prediction": response_dict,
        "latency": 1000 * (time.monotonic() - start_time),  # in ms
    }
    logger.info(log_data)
    return response_dict


@app.get("/")
def read_root():
    return {"Hello": "World"}
