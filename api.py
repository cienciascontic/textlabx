# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sklearn.linear_model import LogisticRegression
import os
import uuid
import requests

app = FastAPI()

# CORS (podés restringirlo luego; por ahora abierto)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "modelos"
os.makedirs(MODEL_DIR, exist_ok=True)

model_cache = {}

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


class Ejemplo(BaseModel):
    texto: str
    categoria: str


class DatosEntrenamiento(BaseModel):
    ejemplos: list[Ejemplo]


class TextoEntrada(BaseModel):
    texto: str


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN", "")
    return token.strip()


def embed_texts(texts):

    token = os.getenv("HF_TOKEN")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    embeddings = []

    for text in texts:

        payload = {
            "inputs": text
        }

        r = requests.post(HF_API_URL, headers=headers, json=payload)

        if not r.ok:
            raise RuntimeError(f"HuggingFace error: {r.text}")

        data = r.json()

        # si devuelve token embeddings
        if isinstance(data[0], list):
            avg = [sum(col)/len(col) for col in zip(*data)]
            embeddings.append(avg)

        else:
            embeddings.append(data)

    return embeddings

@app.get("/")
def home():
    return {"status": "ok", "message": "TextLabX API lista (HF embeddings)"}


@app.post("/train")
def train(data: DatosEntrenamiento):
    textos = [e.texto for e in data.ejemplos]
    labels = [e.categoria for e in data.ejemplos]

    # Embeddings semánticos via HF
    X = embed_texts(textos)

    # Clasificador liviano
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    model_id = uuid.uuid4().hex[:8]
    model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")

    joblib.dump(clf, model_path)
    model_cache[model_id] = clf

    return {"status": "ok", "model_id": model_id, "endpoint": f"/predict/{model_id}"}


@app.post("/predict/{model_id}")
def predict(model_id: str, data: TextoEntrada):
    if model_id in model_cache:
        model = model_cache[model_id]
    else:
        model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")
        if not os.path.exists(model_path):
            return {"error": "Modelo no encontrado. Verificá el ID."}
        model = joblib.load(model_path)
        model_cache[model_id] = model

    vec = embed_texts([data.texto])
    categoria = model.predict(vec)[0]
    return {"categoria": categoria}
