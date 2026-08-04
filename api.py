# api.py
#
# Versión corregida del api.py real del repo https://github.com/cienciascontic/textlabx
# (el que efectivamente corre en textlabx.onrender.com). Pensado para reemplazar
# ese archivo 1:1 — mismos endpoints, mismo contrato con el frontend y la extensión.
#
# Cambios respecto del original, de mayor a menor impacto:
#
#   1) BUG CRÍTICO en embed_texts(): el payload que se mandaba a HuggingFace
#      ({"inputs": {"source_sentence": texto, "sentences": [texto]}}) dispara la
#      tarea de *sentence-similarity*, no *feature-extraction*. Eso hace que la
#      API devuelva un score de similitud del texto contra sí mismo (~1.0 siempre),
#      no un embedding. Resultado: todas las frases terminaban representadas por
#      (casi) el mismo número, sin importar su contenido, y el clasificador
#      aprendía a devolver siempre la misma categoría. Se corrige a
#      {"inputs": texto}, que es el payload correcto de feature-extraction.
#
#   2) /train ahora valida los datos antes de entrenar (mínimo de categorías y de
#      ejemplos por categoría) y devuelve un diagnóstico de calidad (accuracy sobre
#      el propio training set + qué categorías el modelo no logra distinguir) en
#      vez de responder "ok" ciegamente sin importar si el modelo sirve o no.
#
#   3) /predict devuelve un 404 real (HTTPException) cuando el model_id no existe,
#      en vez de {"error": ...} con status 200 — antes eso se colaba como una
#      respuesta "exitosa" en el frontend y mostraba un mensaje genérico que no
#      explicaba nada.
#
#   4) Se agrega un "boot_id" (un ID aleatorio generado en cada arranque del
#      proceso). Como Render tiene disco efímero, cualquier modelo entrenado antes
#      de un reinicio deja de existir. Con el boot_id el frontend puede detectar
#      ese reinicio de forma proactiva (comparando el boot_id con el que tenía el
#      servidor al entrenar/compartir el modelo) en vez de que el usuario se entere
#      recién cuando falla una predicción.
#
#   5) get_hf_token() (que ya existía pero nunca se usaba) ahora sí se usa, para
#      que un HF_TOKEN con espacios/saltos de línea de más (típico al pegarlo en
#      el panel de Render) no rompa la autenticación en silencio.
#
#   6) Los errores al llamar a HuggingFace (token inválido, timeout, modelo
#      cargando en frío, HF caído) ahora se distinguen y se devuelven como un 502
#      con el motivo real, en vez de un 500 genérico sin explicación.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
import os
import time
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
HF_TIMEOUT_SEGUNDOS = 30

# --- Detección de reinicios del servidor (disco efímero en hosts como Render) ---
# Cada arranque del proceso genera un ID nuevo. Los modelos guardados en disco o
# en la caché en memoria de un arranque anterior ya no existen, así que si el
# boot_id cambió, cualquier model_id previo es inválido por definición.
SERVER_BOOT_ID = uuid.uuid4().hex
SERVER_STARTED_AT = time.time()

# --- Umbrales de validación de calidad del entrenamiento ---
MIN_CATEGORIAS = 2               # hacen falta al menos 2 categorías para clasificar
MIN_EJEMPLOS_POR_CATEGORIA = 3   # con menos, el modelo no tiene con qué generalizar
UMBRAL_ACCURACY_ACEPTABLE = 0.7  # por debajo de esto, avisamos que puede no ser confiable


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
    """
    Pide embeddings semánticos reales (tarea feature-extraction) a la Inference
    API de HuggingFace para sentence-transformers/all-MiniLM-L6-v2.

    El payload correcto para feature-extraction es {"inputs": texto}. Un payload
    tipo {"inputs": {"source_sentence": ..., "sentences": [...]}} dispara la tarea
    de *sentence-similarity* en cambio, y devuelve un score de similitud (un solo
    número) en vez de un vector de embedding — ese fue el bug que hacía que el
    modelo entrenado predijera siempre la misma categoría.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "Falta configurar la variable de entorno HF_TOKEN en el servidor "
            "(o está vacía). Sin un token válido de HuggingFace no se pueden "
            "calcular embeddings."
        )

    headers = {"Authorization": f"Bearer {token}"}
    embeddings = []

    for text in texts:
        payload = {"inputs": text}

        try:
            r = requests.post(
                HF_API_URL, headers=headers, json=payload, timeout=HF_TIMEOUT_SEGUNDOS
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"HuggingFace no respondió en {HF_TIMEOUT_SEGUNDOS}s al pedir un "
                f"embedding. El modelo '{HF_MODEL}' puede estar cargando en frío; "
                f"probá de nuevo en un rato."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"No se pudo conectar con HuggingFace: {e}")

        if r.status_code == 401:
            raise RuntimeError(
                "HuggingFace rechazó el token (401). Verificá que HF_TOKEN esté "
                "bien configurado en el servidor y no haya expirado."
            )
        if r.status_code == 503:
            raise RuntimeError(
                f"El modelo '{HF_MODEL}' está cargando en HuggingFace (503). "
                f"Reintentá en unos segundos."
            )
        if not r.ok:
            raise RuntimeError(f"Error de HuggingFace ({r.status_code}): {r.text}")

        data = r.json()

        # Según el backend puede devolver:
        #  - un vector plano ya pooleado: [0.01, -0.23, ...]
        #  - embeddings por token (lista de listas): se promedian ("mean pooling")
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            avg = [sum(col) / len(col) for col in zip(*data)]
            embeddings.append(avg)
        else:
            embeddings.append(data)

    return embeddings


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "TextLabX API lista (HF embeddings)",
        "boot_id": SERVER_BOOT_ID,
        "started_at": SERVER_STARTED_AT
    }


@app.post("/train")
def train(data: DatosEntrenamiento):
    textos = [e.texto for e in data.ejemplos]
    labels = [e.categoria for e in data.ejemplos]

    conteo_por_categoria = dict(Counter(labels))

    # --- Validaciones previas: sin esto, el modelo se entrena "a ciegas" ---
    if len(conteo_por_categoria) < MIN_CATEGORIAS:
        return {
            "status": "error",
            "error": (
                f"Hacen falta al menos {MIN_CATEGORIAS} categorías distintas con ejemplos "
                f"para poder entrenar un clasificador. Encontradas: {len(conteo_por_categoria)}."
            ),
            "conteo_por_categoria": conteo_por_categoria
        }

    categorias_insuficientes = [
        c for c, n in conteo_por_categoria.items() if n < MIN_EJEMPLOS_POR_CATEGORIA
    ]
    if categorias_insuficientes:
        return {
            "status": "error",
            "error": (
                f"Las categorías {', '.join(categorias_insuficientes)} tienen menos de "
                f"{MIN_EJEMPLOS_POR_CATEGORIA} ejemplos. Con tan pocos ejemplos el modelo "
                f"no tiene suficiente información para distinguirlas de las demás. "
                f"Agregá más frases antes de entrenar."
            ),
            "conteo_por_categoria": conteo_por_categoria
        }

    # Embeddings semánticos vía HF
    try:
        X = embed_texts(textos)
    except RuntimeError as e:
        # Antes esto explotaba como un 500 sin explicación. Ahora se distingue de
        # otros errores del servidor y se explica la causa real.
        raise HTTPException(status_code=502, detail=str(e))

    # class_weight="balanced" evita que una categoría con muchos más ejemplos que
    # las otras termine "tapando" a las categorías minoritarias en la predicción.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)

    # --- Diagnóstico de calidad: ¿el modelo aprendió algo razonable? ---
    predicciones_train = clf.predict(X)
    train_accuracy = accuracy_score(labels, predicciones_train)
    clases_nunca_predichas = sorted(set(labels) - set(predicciones_train))

    advertencias = []
    if train_accuracy < UMBRAL_ACCURACY_ACEPTABLE:
        advertencias.append(
            f"El modelo solo clasifica correctamente el {train_accuracy:.0%} de sus propios "
            f"ejemplos de entrenamiento. Es probable que las frases de las distintas categorías "
            f"se parezcan demasiado entre sí (semánticamente). Agregá ejemplos más variados y "
            f"distintivos."
        )
    if clases_nunca_predichas:
        advertencias.append(
            "Las categorías " + ", ".join(clases_nunca_predichas) + " nunca son predichas, "
            "ni siquiera para sus propios ejemplos de entrenamiento: otra categoría las está "
            "'tapando'. Agregá más ejemplos distintivos para esas categorías."
        )

    model_id = uuid.uuid4().hex[:8]
    model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")

    joblib.dump(clf, model_path)
    model_cache[model_id] = clf

    return {
        "status": "ok",
        "model_id": model_id,
        "endpoint": f"/predict/{model_id}",
        "train_accuracy": round(train_accuracy, 4),
        "conteo_por_categoria": conteo_por_categoria,
        "advertencias": advertencias,
        "boot_id": SERVER_BOOT_ID
    }


@app.post("/predict/{model_id}")
def predict(model_id: str, data: TextoEntrada):
    if model_id in model_cache:
        model = model_cache[model_id]
    else:
        model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")
        if not os.path.exists(model_path):
            # 404 real: antes devolvía {"error": ...} con status 200, lo que hacía
            # que el frontend lo tratara como una respuesta exitosa y mostrara un
            # mensaje genérico en vez de explicar que el modelo no existe.
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{model_id}' no encontrado. Puede que el servidor se haya "
                       f"reiniciado y haya perdido los modelos entrenados, o que el ID sea "
                       f"incorrecto."
            )
        model = joblib.load(model_path)
        model_cache[model_id] = model

    try:
        vec = embed_texts([data.texto])
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    categoria = model.predict(vec)[0]
    return {"categoria": categoria, "boot_id": SERVER_BOOT_ID}
