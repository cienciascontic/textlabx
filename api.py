# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import os
import uuid

app = FastAPI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta donde guardamos todos los modelos
MODEL_DIR = "modelos"
os.makedirs(MODEL_DIR, exist_ok=True)

# Cache simple en memoria {model_id: model}
model_cache = {}


class Ejemplo(BaseModel):
    texto: str
    categoria: str

class DatosEntrenamiento(BaseModel):
    ejemplos: list[Ejemplo]

class TextoEntrada(BaseModel):
    texto: str


@app.get("/")
def home():
    return {"status": "ok", "message": "TextLabX API lista"}


@app.post("/train")
def train(data: DatosEntrenamiento):
    """
    Recibe:
    {
      "ejemplos": [
        {"texto": "...", "categoria": "..."},
        ...
      ]
    }
    Entrena un modelo nuevo, lo guarda como .pkl y devuelve un ID.
    """

    textos = [e.texto for e in data.ejemplos]
    labels = [e.categoria for e in data.ejemplos]

 # convertir textos a embeddings semánticos
X = embedder.encode(textos)

# clasificador
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

model_local = clf

    model_local.fit(textos, labels)

    # ID único para este modelo
    model_id = uuid.uuid4().hex[:8]
    model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")

    joblib.dump(model_local, model_path)
    model_cache[model_id] = model_local

    return {
        "status": "ok",
        "model_id": model_id,
        "endpoint": f"/predict/{model_id}"
    }


@app.post("/predict/{model_id}")
def predict(model_id: str, data: TextoEntrada):
    """
    POST /predict/{model_id}
    body: { "texto": "frase a clasificar" }
    """

    # Buscar en cache
    if model_id in model_cache:
        model = model_cache[model_id]
    else:
        model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")
        if not os.path.exists(model_path):
            return {"error": "Modelo no encontrado. Verificá el ID."}
        model = joblib.load(model_path)
        model_cache[model_id] = model

   vec = embedder.encode([data.texto])
categoria = model.predict(vec)[0]
    return {"categoria": categoria}
