# api.py
#
# Versión corregida (v2) del api.py real del repo https://github.com/cienciascontic/textlabx
# (el que corre en textlabx.onrender.com). Reemplaza 1:1 al archivo actual del repo —
# mismos endpoints, mismo contrato con el frontend y la extensión de Scratch.
#
# ¿Por qué v2? La v1 de este fix cambiaba el payload de embed_texts() para pedir
# "feature-extraction" en vez de "sentence-similarity", asumiendo que el problema
# era solo el formato del payload. Al probarlo contra el servidor real, HuggingFace
# devolvió: "SentenceSimilarityPipeline.__call__() missing 1 required positional
# argument: 'sentences'". Investigando la metadata pública del modelo confirmamos
# que sentence-transformers/all-MiniLM-L6-v2 —y en general toda la familia
# sentence-transformers— en el proveedor "hf-inference" que se está usando SOLO
# tiene registrada la tarea sentence-similarity. No existe combinación de payload
# que le saque un vector de feature-extraction: esa tarea no está habilitada para
# este modelo en ese proveedor, así que la arquitectura "pedir un embedding y
# entrenar un LogisticRegression con esos vectores" no es viable tal como estaba.
#
# REDISEÑO: en vez de generar embeddings y entrenar un clasificador, este archivo:
#   - /train guarda las frases + categorías tal cual (ninguna llamada a HF; entrenar
#     es instantáneo y no puede fallar por HuggingFace).
#   - /predict hace UNA sola llamada a HF usando sentence-similarity (la tarea que
#     sí funciona con este modelo): compara la frase nueva contra TODOS los
#     ejemplos guardados en una sola request, y devuelve la categoría del ejemplo
#     más parecido semánticamente (1-vecino-más-cercano por categoría).
#
# Otros fixes que se mantienen de la v1:
#   - /predict devuelve 404 real (HTTPException) si el model_id no existe.
#   - boot_id / started_at para detectar reinicios del servidor (disco efímero
#     de Render) desde el frontend, de forma proactiva.
#   - get_hf_token() se usa de verdad (con .strip()) para evitar tokens con
#     espacios/saltos de línea de más.
#   - Errores de HuggingFace (token inválido, timeout, modelo cargando en frío,
#     HF caído) se devuelven como 502 con el motivo real, no como un 500 pelado.
#   - /train valida categorías/ejemplos mínimos y detecta si dos categorías
#     comparten una frase idéntica (un error de carga típico) antes de aceptar
#     el entrenamiento.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from collections import Counter, defaultdict
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
MIN_EJEMPLOS_POR_CATEGORIA = 2   # con menos, no hay nada contra qué comparar


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


def _post_hf(payload: dict) -> dict:
    """POST genérico a la Inference API de HF, con manejo de errores claro."""
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "Falta configurar la variable de entorno HF_TOKEN en el servidor "
            "(o está vacía). Sin un token válido de HuggingFace no se puede clasificar."
        )

    headers = {"Authorization": f"Bearer {token}"}

    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=HF_TIMEOUT_SEGUNDOS)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"HuggingFace no respondió en {HF_TIMEOUT_SEGUNDOS}s. El modelo "
            f"'{HF_MODEL}' puede estar cargando en frío; probá de nuevo en un rato."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con HuggingFace: {e}")

    if r.status_code == 401:
        raise RuntimeError(
            "HuggingFace rechazó el token (401). Verificá que HF_TOKEN esté bien "
            "configurado en el servidor y no haya expirado."
        )
    if r.status_code == 503:
        raise RuntimeError(
            f"El modelo '{HF_MODEL}' está cargando en HuggingFace (503). Reintentá "
            f"en unos segundos."
        )
    if not r.ok:
        raise RuntimeError(f"Error de HuggingFace ({r.status_code}): {r.text}")

    return r.json()


def categoria_mas_similar(texto_nuevo: str, ejemplos: list[dict]):
    """
    Compara `texto_nuevo` contra TODOS los ejemplos guardados en una sola llamada
    a la tarea sentence-similarity de HF (la única que este modelo soporta en el
    proveedor hf-inference), y devuelve la categoría del ejemplo más parecido
    semánticamente, junto con el score de esa mejor coincidencia por categoría.
    """
    textos_candidatos = [e["texto"] for e in ejemplos]

    payload = {
        "inputs": {
            "source_sentence": texto_nuevo,
            "sentences": textos_candidatos
        }
    }
    scores = _post_hf(payload)  # lista de floats, mismo orden que textos_candidatos

    mejor_score_por_categoria = defaultdict(lambda: -1.0)
    for ejemplo, score in zip(ejemplos, scores):
        cat = ejemplo["categoria"]
        if score > mejor_score_por_categoria[cat]:
            mejor_score_por_categoria[cat] = score

    categoria_ganadora = max(mejor_score_por_categoria, key=mejor_score_por_categoria.get)
    return categoria_ganadora, dict(mejor_score_por_categoria)


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
    ejemplos = [{"texto": e.texto, "categoria": e.categoria} for e in data.ejemplos]
    labels = [e["categoria"] for e in ejemplos]

    conteo_por_categoria = dict(Counter(labels))

    # --- Validaciones: sin esto, "entrenar" acepta cualquier cosa a ciegas ---
    if len(conteo_por_categoria) < MIN_CATEGORIAS:
        return {
            "status": "error",
            "error": (
                f"Hacen falta al menos {MIN_CATEGORIAS} categorías distintas con ejemplos "
                f"para poder clasificar. Encontradas: {len(conteo_por_categoria)}."
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
                f"{MIN_EJEMPLOS_POR_CATEGORIA} ejemplos. Agregá más frases antes de entrenar."
            ),
            "conteo_por_categoria": conteo_por_categoria
        }

    # Detecta un error de carga común: la misma frase exacta cargada bajo dos
    # categorías distintas (confunde cualquier método de clasificación).
    categorias_por_texto = defaultdict(set)
    for e in ejemplos:
        categorias_por_texto[e["texto"].strip().lower()].add(e["categoria"])
    frases_ambiguas = [t for t, cats in categorias_por_texto.items() if len(cats) > 1]

    advertencias = []
    if frases_ambiguas:
        advertencias.append(
            "Estas frases están cargadas en más de una categoría a la vez, lo cual "
            "las hace imposibles de distinguir: " + "; ".join(f'"{f}"' for f in frases_ambiguas)
        )

    # No hay "entrenamiento" real que pueda fallar por HuggingFace: guardamos los
    # ejemplos tal cual. La comparación semántica se hace en el momento de predecir.
    model_id = uuid.uuid4().hex[:8]
    model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")

    modelo_guardado = {"tipo": "similaridad-hf-v1", "ejemplos": ejemplos}
    joblib.dump(modelo_guardado, model_path)
    model_cache[model_id] = modelo_guardado

    return {
        "status": "ok",
        "model_id": model_id,
        "endpoint": f"/predict/{model_id}",
        "conteo_por_categoria": conteo_por_categoria,
        "advertencias": advertencias,
        "boot_id": SERVER_BOOT_ID
    }


@app.post("/predict/{model_id}")
def predict(model_id: str, data: TextoEntrada):
    if model_id in model_cache:
        modelo = model_cache[model_id]
    else:
        model_path = os.path.join(MODEL_DIR, f"modelo_{model_id}.pkl")
        if not os.path.exists(model_path):
            # 404 real: antes devolvía {"error": ...} con status 200, lo que hacía
            # que el frontend lo tratara como éxito y mostrara un mensaje genérico
            # en vez de explicar que el modelo no existe.
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{model_id}' no encontrado. Puede que el servidor se haya "
                       f"reiniciado y haya perdido los modelos entrenados, o que el ID sea "
                       f"incorrecto."
            )
        modelo = joblib.load(model_path)
        model_cache[model_id] = modelo

    # Modelos entrenados con una versión anterior (LogisticRegression + embeddings)
    # no son compatibles con este formato nuevo: son inservibles de todos modos,
    # porque esa versión anterior tenía el bug que predecía siempre la misma
    # categoría. Mejor avisar claro que fallar con un error críptico.
    if not (isinstance(modelo, dict) and modelo.get("tipo") == "similaridad-hf-v1"):
        raise HTTPException(
            status_code=410,
            detail=f"El modelo '{model_id}' fue entrenado con una versión anterior "
                   f"del sistema y ya no es compatible. Volvé a entrenarlo."
        )

    try:
        categoria, scores_por_categoria = categoria_mas_similar(data.texto, modelo["ejemplos"])
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "categoria": categoria,
        "similitud": round(scores_por_categoria[categoria], 4),
        "boot_id": SERVER_BOOT_ID
    }
