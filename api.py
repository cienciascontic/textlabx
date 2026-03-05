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
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"


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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Devuelve embeddings (lista de vectores) usando Hugging Face Inference API.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError("Falta HF_TOKEN en variables de entorno (Render).")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")

    data = r.json()

    # Para este endpoint normalmente devuelve una lista de embeddings (uno por texto)
    # Si HF devuelve un embedding 2D/3D raro, lo normalizamos acá.
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and isinstance(data[0][0], (int, float)):
        # ya es [ [384], [384], ... ]
        return data

    # Si viene algo diferente, lo rechazamos explícitamente
    raise RuntimeError("Respuesta inesperada de HF (estructura no reconocida).")


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
