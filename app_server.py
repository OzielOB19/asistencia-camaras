from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from supabase_config import supabase


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODELO_YUNET = BASE_DIR / "modelos" / "face_detection_yunet_2023mar.onnx"
MODELO_SFACE = BASE_DIR / "modelos" / "face_recognition_sface_2021dec.onnx"

DATOS_DIR = BASE_DIR / "datos_sface"
TEMP_DIR = BASE_DIR / "registro_temporal"

BUCKET = os.getenv("SUPABASE_BUCKET", "rostros").strip()
TABLA = os.getenv("SUPABASE_TABLE", "perfiles_sface").strip()

ETAPAS = ["FRENTE", "IZQUIERDA", "DERECHA", "ARRIBA", "ABAJO"]
FOTOS_POR_ETAPA = 10
TOTAL_FOTOS = len(ETAPAS) * FOTOS_POR_ETAPA

MIN_DETECCION_REGISTRO = float(
    os.getenv("MIN_DETECCION_REGISTRO", "0.70")
)
MIN_DETECCION_RECONOCIMIENTO = float(
    os.getenv("MIN_DETECCION_RECONOCIMIENTO", "0.50")
)
MIN_DETECCION_SEGUIMIENTO = float(
    os.getenv("MIN_DETECCION_SEGUIMIENTO", "0.46")
)
MIN_ROSTRO_REGISTRO = int(
    os.getenv("MIN_ROSTRO_REGISTRO", "90")
)
MIN_ROSTRO_RECONOCIMIENTO = int(
    os.getenv("MIN_ROSTRO_RECONOCIMIENTO", "28")
)
MIN_NITIDEZ = float(os.getenv("MIN_NITIDEZ", "35.0"))
UMBRAL_SFACE = float(os.getenv("UMBRAL_SFACE", "0.40"))
ANCHO_DETECCION_RAPIDA = int(
    os.getenv("ANCHO_DETECCION_RAPIDA", "1120")
)
ANCHO_DETECCION_DETALLE = int(
    os.getenv("ANCHO_DETECCION_DETALLE", "1600")
)
ANCHO_SEGUIMIENTO = int(
    os.getenv("ANCHO_SEGUIMIENTO", "896")
)
MIN_ROSTRO_SEGUIMIENTO = int(
    os.getenv("MIN_ROSTRO_SEGUIMIENTO", "20")
)
OPENCV_THREADS = int(os.getenv("OPENCV_THREADS", "2"))
MAX_IMAGEN = 8 * 1024 * 1024

cv2.setNumThreads(max(1, OPENCV_THREADS))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGEN

DATOS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

detector_registro_lock = threading.Lock()
detector_reconocimiento_lock = threading.Lock()
detector_seguimiento_lock = threading.Lock()
reconocedor_lock = threading.Lock()
archivo_lock = threading.Lock()
galeria_lock = threading.Lock()

_galeria_cache: dict[str, Any] = {
    "firma": None,
    "perfiles": [],
}

if not MODELO_YUNET.exists():
    raise FileNotFoundError(f"No se encontró YuNet: {MODELO_YUNET}")
if not MODELO_SFACE.exists():
    raise FileNotFoundError(f"No se encontró SFace: {MODELO_SFACE}")

detector_registro = cv2.FaceDetectorYN.create(
    str(MODELO_YUNET),
    "",
    (320, 320),
    MIN_DETECCION_REGISTRO,
    0.30,
    5000,
)

detector_reconocimiento = cv2.FaceDetectorYN.create(
    str(MODELO_YUNET),
    "",
    (320, 320),
    MIN_DETECCION_RECONOCIMIENTO,
    0.30,
    5000,
)

detector_seguimiento = cv2.FaceDetectorYN.create(
    str(MODELO_YUNET),
    "",
    (320, 320),
    MIN_DETECCION_SEGUIMIENTO,
    0.30,
    5000,
)

reconocedor = cv2.FaceRecognizerSF.create(
    str(MODELO_SFACE),
    "",
)


def respuesta_error(mensaje: str, status: int = 400, **extra: Any):
    datos = {"ok": False, "mensaje": mensaje}
    datos.update(extra)
    return jsonify(datos), status


def limpiar_nombre(nombre: str) -> str:
    nombre = re.sub(
        r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ_-]",
        "_",
        nombre.strip(),
    )
    nombre = re.sub(r"_+", "_", nombre)
    return nombre.strip("_")


def validar_sesion(sesion: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,120}", sesion):
        raise ValueError("Sesión inválida")


def sesion_dir(sesion: str) -> Path:
    validar_sesion(sesion)
    return TEMP_DIR / sesion


def meta_path(sesion: str) -> Path:
    return sesion_dir(sesion) / "metadata.json"


def leer_meta(sesion: str) -> dict[str, Any]:
    ruta = meta_path(sesion)
    if not ruta.exists():
        raise FileNotFoundError("La sesión no existe o ya terminó")

    with archivo_lock:
        return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_meta(sesion: str, datos: dict[str, Any]) -> None:
    with archivo_lock:
        meta_path(sesion).write_text(
            json.dumps(datos, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def etapa_pendiente(meta: dict[str, Any]) -> str | None:
    for etapa in ETAPAS:
        if int(meta["conteos"].get(etapa, 0)) < FOTOS_POR_ETAPA:
            return etapa
    return None


def normalizar_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norma = float(np.linalg.norm(vector))
    return vector / norma if norma > 0 else vector


def normalizar_matriz(matriz: np.ndarray) -> np.ndarray:
    matriz = np.asarray(matriz, dtype=np.float32)

    if matriz.ndim == 1:
        matriz = matriz.reshape(1, -1)

    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


def leer_imagen_peticion() -> np.ndarray:
    archivo = request.files.get("image")

    if archivo is None:
        raise ValueError("No se recibió la imagen")

    contenido = archivo.read()

    if not contenido:
        raise ValueError("La imagen está vacía")
    if len(contenido) > MAX_IMAGEN:
        raise ValueError("La imagen es demasiado grande")

    frame = cv2.imdecode(
        np.frombuffer(contenido, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None or frame.size == 0:
        raise ValueError("No se pudo leer la imagen")

    return frame


def detectar_caras_con(
    frame: np.ndarray,
    detector_obj,
    lock: threading.Lock,
) -> np.ndarray | None:
    alto, ancho = frame.shape[:2]
    with lock:
        detector_obj.setInputSize((ancho, alto))
        _, caras = detector_obj.detect(frame)
    return caras


def detectar_caras_registro(
    frame: np.ndarray,
) -> np.ndarray | None:
    return detectar_caras_con(
        frame,
        detector_registro,
        detector_registro_lock,
    )


def _detectar_reconocimiento_en_ancho(
    frame: np.ndarray,
    ancho_objetivo: int,
) -> np.ndarray | None:
    """
    Ejecuta YuNet en una copia reducida para evitar procesar todo el
    fotograma 1080p/2K. Después devuelve cajas y landmarks escalados a
    las coordenadas del frame original, para que SFace recorte desde la
    imagen de mayor calidad.
    """
    alto_original, ancho_original = frame.shape[:2]

    if ancho_original <= ancho_objetivo:
        frame_detector = frame
        factor = 1.0
    else:
        factor_reduccion = ancho_objetivo / ancho_original
        alto_detector = max(1, int(round(alto_original * factor_reduccion)))
        frame_detector = cv2.resize(
            frame,
            (ancho_objetivo, alto_detector),
            interpolation=cv2.INTER_AREA,
        )
        factor = ancho_original / ancho_objetivo

    caras = detectar_caras_con(
        frame_detector,
        detector_reconocimiento,
        detector_reconocimiento_lock,
    )

    if caras is None or factor == 1.0:
        return caras

    caras_original = caras.copy()

    # YuNet: bbox (4) + cinco landmarks (10) + score (1).
    # Solo se escalan las coordenadas; el score queda intacto.
    caras_original[:, :14] *= factor
    return caras_original


def detectar_caras_reconocimiento(
    frame: np.ndarray,
    modo_detalle: bool = False,
) -> tuple[np.ndarray | None, str]:
    """
    Modo rápido para la mayoría de cuadros y un barrido de detalle
    periódico para recuperar caras pequeñas/lejanos sin trabar la UI.
    """
    ancho = (
        ANCHO_DETECCION_DETALLE
        if modo_detalle
        else ANCHO_DETECCION_RAPIDA
    )

    caras = _detectar_reconocimiento_en_ancho(
        frame,
        ancho,
    )

    if (
        not modo_detalle
        and (caras is None or len(caras) == 0)
        and ANCHO_DETECCION_DETALLE > ANCHO_DETECCION_RAPIDA
    ):
        caras = _detectar_reconocimiento_en_ancho(
            frame,
            ANCHO_DETECCION_DETALLE,
        )
        return caras, "detalle_reintento"

    return caras, ("detalle" if modo_detalle else "rapido")


def detectar_seguimiento_en_frame(
    frame: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Ruta rápida para mover los recuadros. Solo ejecuta YuNet:
    no carga galería y no ejecuta SFace.
    """
    alto_original, ancho_original = frame.shape[:2]

    if ancho_original <= ANCHO_SEGUIMIENTO:
        frame_detector = frame
        factor = 1.0
    else:
        factor_reduccion = ANCHO_SEGUIMIENTO / ancho_original
        alto_detector = max(
            1,
            int(round(alto_original * factor_reduccion)),
        )
        frame_detector = cv2.resize(
            frame,
            (ANCHO_SEGUIMIENTO, alto_detector),
            interpolation=cv2.INTER_AREA,
        )
        factor = ancho_original / ANCHO_SEGUIMIENTO

    caras = detectar_caras_con(
        frame_detector,
        detector_seguimiento,
        detector_seguimiento_lock,
    )

    if caras is None:
        return []

    resultados: list[dict[str, Any]] = []

    for cara in caras:
        x, y, ancho, alto = cara[:4].astype(float)
        score = float(cara[-1])

        x *= factor
        y *= factor
        ancho *= factor
        alto *= factor

        if (
            ancho < MIN_ROSTRO_SEGUIMIENTO
            or alto < MIN_ROSTRO_SEGUIMIENTO
        ):
            continue

        x_i = max(0, min(int(round(x)), ancho_original - 1))
        y_i = max(0, min(int(round(y)), alto_original - 1))
        w_i = max(
            1,
            min(int(round(ancho)), ancho_original - x_i),
        )
        h_i = max(
            1,
            min(int(round(alto)), alto_original - y_i),
        )

        resultados.append(
            {
                "score": round(score, 4),
                "caja": {
                    "x": x_i,
                    "y": y_i,
                    "ancho": w_i,
                    "alto": h_i,
                },
            }
        )

    return resultados

def extraer_rostro_registro(
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    caras = detectar_caras_registro(frame)
    cantidad = 0 if caras is None else len(caras)

    if cantidad == 0:
        raise ValueError("No se detecta ninguna cara")
    if cantidad > 1:
        raise ValueError("Debe aparecer solamente una persona")

    cara = caras[0]
    _, _, ancho, alto = cara[:4].astype(int)

    if ancho < MIN_ROSTRO_REGISTRO or alto < MIN_ROSTRO_REGISTRO:
        raise ValueError("Acércate un poco más a la cámara")

    with reconocedor_lock:
        rostro = reconocedor.alignCrop(frame, cara)

        if rostro is None or rostro.size == 0:
            raise ValueError("No se pudo alinear el rostro")

        gris = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())

        if nitidez < MIN_NITIDEZ:
            raise ValueError(
                "La imagen está borrosa; mantén la cara quieta"
            )

        embedding = normalizar_vector(
            reconocedor.feature(rostro)
        )

    return rostro, embedding, nitidez


def nombre_desde_npz(datos: Any, ruta: Path) -> str:
    if "nombre" not in datos.files:
        return ruta.parent.parent.name

    valor = datos["nombre"]

    try:
        if np.asarray(valor).shape == ():
            nombre = str(np.asarray(valor).item())
        else:
            nombre = str(np.asarray(valor).reshape(-1)[0])
    except (ValueError, TypeError, IndexError):
        nombre = ruta.parent.parent.name

    nombre = nombre.strip()
    return nombre or ruta.parent.parent.name


def firma_galeria() -> tuple[tuple[str, int, int], ...]:
    firma = []

    for ruta in sorted(DATOS_DIR.glob("*/*/embeddings.npz")):
        try:
            stat = ruta.stat()
            firma.append(
                (
                    str(ruta.relative_to(DATOS_DIR)),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
        except FileNotFoundError:
            continue

    return tuple(firma)


def cargar_galeria(forzar: bool = False) -> list[dict[str, Any]]:
    firma = firma_galeria()

    with galeria_lock:
        if (
            not forzar
            and _galeria_cache["firma"] == firma
        ):
            return _galeria_cache["perfiles"]

        agrupados: dict[str, dict[str, Any]] = {}

        for ruta in sorted(
            DATOS_DIR.glob("*/*/embeddings.npz")
        ):
            try:
                with np.load(
                    ruta,
                    allow_pickle=False,
                ) as datos:
                    if "embeddings" not in datos.files:
                        continue

                    embeddings = normalizar_matriz(
                        datos["embeddings"]
                    )
                    nombre = nombre_desde_npz(datos, ruta)

                if embeddings.size == 0:
                    continue

                clave = nombre.casefold()

                if clave not in agrupados:
                    agrupados[clave] = {
                        "nombre": nombre,
                        "bloques": [],
                        "sesiones": 0,
                    }

                agrupados[clave]["bloques"].append(
                    embeddings
                )
                agrupados[clave]["sesiones"] += 1

            except (
                OSError,
                ValueError,
                KeyError,
            ) as exc:
                print(
                    f"[GALERÍA] Se omitió {ruta}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        perfiles: list[dict[str, Any]] = []

        for datos in agrupados.values():
            matriz = normalizar_matriz(
                np.vstack(datos["bloques"])
            )

            perfiles.append(
                {
                    "nombre": datos["nombre"],
                    "embeddings": matriz,
                    "muestras": int(len(matriz)),
                    "sesiones": int(datos["sesiones"]),
                }
            )

        perfiles.sort(
            key=lambda perfil: perfil["nombre"].casefold()
        )

        _galeria_cache["firma"] = firma
        _galeria_cache["perfiles"] = perfiles
        return perfiles


def invalidar_galeria() -> None:
    with galeria_lock:
        _galeria_cache["firma"] = None
        _galeria_cache["perfiles"] = []


def mejor_coincidencia(
    embedding: np.ndarray,
    perfiles: list[dict[str, Any]],
) -> tuple[str, float]:
    mejor_nombre = "desconocido"
    mejor_similitud = -1.0

    consulta = normalizar_vector(embedding)

    for perfil in perfiles:
        similitudes = perfil["embeddings"] @ consulta

        if similitudes.size == 0:
            continue

        cantidad = min(3, int(similitudes.size))
        mejores = np.partition(
            similitudes,
            -cantidad,
        )[-cantidad:]

        puntuacion = float(np.mean(mejores))

        if puntuacion > mejor_similitud:
            mejor_similitud = puntuacion
            mejor_nombre = perfil["nombre"]

    if mejor_similitud < UMBRAL_SFACE:
        mejor_nombre = "desconocido"

    return mejor_nombre, mejor_similitud


def reconocer_en_frame(
    frame: np.ndarray,
    modo_detalle: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    perfiles = cargar_galeria()
    resultados: list[dict[str, Any]] = []

    caras, modo_usado = detectar_caras_reconocimiento(
        frame,
        modo_detalle=modo_detalle,
    )

    if caras is None:
        return resultados, modo_usado

    alto_frame, ancho_frame = frame.shape[:2]

    for cara in caras:
        cara_original = cara.copy()
        x, y, ancho, alto = cara_original[:4].astype(int)

        if (
            ancho < MIN_ROSTRO_RECONOCIMIENTO
            or alto < MIN_ROSTRO_RECONOCIMIENTO
        ):
            continue

        with reconocedor_lock:
            rostro = reconocedor.alignCrop(
                frame,
                cara_original,
            )

            if rostro is None or rostro.size == 0:
                continue

            embedding = normalizar_vector(
                reconocedor.feature(rostro)
            )

        nombre, similitud = mejor_coincidencia(
            embedding,
            perfiles,
        )

        x = max(0, min(x, ancho_frame - 1))
        y = max(0, min(y, alto_frame - 1))
        ancho = max(1, min(ancho, ancho_frame - x))
        alto = max(1, min(alto, alto_frame - y))

        resultados.append(
            {
                "nombre": nombre,
                "similitud": round(similitud, 4),
                "caja": {
                    "x": int(x),
                    "y": int(y),
                    "ancho": int(ancho),
                    "alto": int(alto),
                },
            }
        )

    return resultados, modo_usado

def subir_archivo(
    local: Path,
    remoto: str,
    content_type: str,
) -> None:
    supabase.storage.from_(BUCKET).upload(
        path=remoto,
        file=local.read_bytes(),
        file_options={
            "content-type": content_type,
            "upsert": "true",
        },
    )


def sincronizar_supabase(
    nombre: str,
    nombre_seguro: str,
    sesion_final: str,
    carpeta_final: Path,
    cantidad: int,
) -> str:
    prefijo = f"sface/{nombre_seguro}/{sesion_final}"
    ruta_npz = f"{prefijo}/embeddings.npz"

    subir_archivo(
        carpeta_final / "embeddings.npz",
        ruta_npz,
        "application/octet-stream",
    )

    for imagen in sorted(
        (carpeta_final / "imagenes").glob("*.jpg")
    ):
        subir_archivo(
            imagen,
            f"{prefijo}/imagenes/{imagen.name}",
            "image/jpeg",
        )

    existente = (
        supabase.table(TABLA)
        .select("id")
        .eq("ruta_embeddings", ruta_npz)
        .limit(1)
        .execute()
    )

    fila = {
        "nombre": nombre,
        "sesion": sesion_final,
        "ruta_embeddings": ruta_npz,
        "cantidad_muestras": cantidad,
    }

    if existente.data:
        (
            supabase.table(TABLA)
            .update(fila)
            .eq("id", existente.data[0]["id"])
            .execute()
        )
    else:
        supabase.table(TABLA).insert(fila).execute()

    return ruta_npz


@app.get("/")
def index():
    return render_template(
        "index.html",
        etapas=ETAPAS,
        fotos_por_etapa=FOTOS_POR_ETAPA,
        total_fotos=TOTAL_FOTOS,
    )


@app.get("/api/salud")
def salud():
    perfiles = cargar_galeria()

    return jsonify(
        {
            "ok": True,
            "servicio": "Registro y reconocimiento web SFace",
            "bucket": BUCKET,
            "tabla": TABLA,
            "personas_cargadas": len(perfiles),
            "muestras_cargadas": sum(
                perfil["muestras"]
                for perfil in perfiles
            ),
            "umbral": UMBRAL_SFACE,
            "deteccion_reconocimiento": MIN_DETECCION_RECONOCIMIENTO,
            "rostro_minimo_reconocimiento": MIN_ROSTRO_RECONOCIMIENTO,
            "ancho_deteccion_rapida": ANCHO_DETECCION_RAPIDA,
            "ancho_deteccion_detalle": ANCHO_DETECCION_DETALLE,
            "deteccion_seguimiento": MIN_DETECCION_SEGUIMIENTO,
            "ancho_seguimiento": ANCHO_SEGUIMIENTO,
        }
    )


@app.get("/api/reconocimiento/perfiles")
def perfiles():
    galeria = cargar_galeria(forzar=True)

    return jsonify(
        {
            "ok": True,
            "personas": [
                {
                    "nombre": perfil["nombre"],
                    "muestras": perfil["muestras"],
                    "sesiones": perfil["sesiones"],
                }
                for perfil in galeria
            ],
        }
    )


@app.post("/api/detectar")
def detectar_rapido():
    inicio = time.perf_counter()

    try:
        frame = leer_imagen_peticion()
        resultados = detectar_seguimiento_en_frame(frame)
    except ValueError as exc:
        return respuesta_error(str(exc))

    alto, ancho = frame.shape[:2]
    procesamiento_ms = (time.perf_counter() - inicio) * 1000.0

    return jsonify(
        {
            "ok": True,
            "rostros": resultados,
            "cantidad": len(resultados),
            "frame_ancho": ancho,
            "frame_alto": alto,
            "procesamiento_ms": round(procesamiento_ms, 1),
        }
    )


@app.post("/api/reconocer")
def reconocer():
    inicio = time.perf_counter()
    modo_detalle = str(
        request.form.get("detalle", "0")
    ).strip() in {"1", "true", "True"}

    try:
        frame = leer_imagen_peticion()
        resultados, modo_usado = reconocer_en_frame(
            frame,
            modo_detalle=modo_detalle,
        )
    except ValueError as exc:
        return respuesta_error(str(exc))

    alto, ancho = frame.shape[:2]
    procesamiento_ms = (time.perf_counter() - inicio) * 1000.0

    return jsonify(
        {
            "ok": True,
            "rostros": resultados,
            "cantidad": len(resultados),
            "frame_ancho": ancho,
            "frame_alto": alto,
            "umbral": UMBRAL_SFACE,
            "modo_deteccion": modo_usado,
            "procesamiento_ms": round(procesamiento_ms, 1),
        }
    )


@app.post("/api/registro/iniciar")
def iniciar():
    datos = request.get_json(silent=True) or {}
    nombre = str(datos.get("nombre", "")).strip()
    nombre_seguro = limpiar_nombre(nombre)

    if len(nombre) < 2 or not nombre_seguro:
        return respuesta_error("Escribe un nombre válido")

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    sesion = (
        f"{nombre_seguro}_{fecha}_{uuid.uuid4().hex[:8]}"
    )
    carpeta = sesion_dir(sesion)

    (carpeta / "imagenes").mkdir(
        parents=True,
        exist_ok=False,
    )
    (carpeta / "embeddings").mkdir(
        parents=True,
        exist_ok=False,
    )

    meta = {
        "sesion": sesion,
        "nombre": nombre,
        "nombre_seguro": nombre_seguro,
        "conteos": {
            etapa: 0
            for etapa in ETAPAS
        },
    }

    guardar_meta(sesion, meta)

    return jsonify(
        {
            "ok": True,
            "sesion": sesion,
            "etapa": ETAPAS[0],
            "conteo": 0,
            "total": 0,
        }
    )


@app.post("/api/registro/capturar")
def capturar():
    sesion = str(
        request.form.get("sesion", "")
    ).strip()

    etapa = str(
        request.form.get("etapa", "")
    ).strip().upper()

    try:
        meta = leer_meta(sesion)
    except (ValueError, FileNotFoundError) as exc:
        return respuesta_error(str(exc), 404)

    esperada = etapa_pendiente(meta)

    if esperada is None:
        return respuesta_error(
            "Todas las etapas ya están completas",
            etapa_completa=True,
            registro_completo=True,
        )

    if etapa != esperada:
        return respuesta_error(
            f"La etapa actual es {esperada}",
            etapa=esperada,
        )

    try:
        frame = leer_imagen_peticion()
        rostro, embedding, nitidez = (
            extraer_rostro_registro(frame)
        )
    except ValueError as exc:
        return respuesta_error(str(exc))

    numero_etapa = int(meta["conteos"][etapa]) + 1
    numero_total = (
        sum(
            int(valor)
            for valor in meta["conteos"].values()
        )
        + 1
    )

    base = (
        f"{numero_total:03d}_"
        f"{etapa}_{numero_etapa:02d}"
    )

    carpeta = sesion_dir(sesion)
    ruta_img = carpeta / "imagenes" / f"{base}.jpg"
    ruta_emb = carpeta / "embeddings" / f"{base}.npy"

    if not cv2.imwrite(str(ruta_img), rostro):
        return respuesta_error(
            "No se pudo guardar la imagen",
            500,
        )

    np.save(
        ruta_emb,
        embedding.astype(np.float32),
    )

    meta["conteos"][etapa] = numero_etapa
    guardar_meta(sesion, meta)

    total = sum(
        int(valor)
        for valor in meta["conteos"].values()
    )

    return jsonify(
        {
            "ok": True,
            "mensaje": "Captura aceptada",
            "etapa": etapa,
            "conteo": numero_etapa,
            "total": total,
            "nitidez": round(nitidez, 1),
            "etapa_completa": (
                numero_etapa >= FOTOS_POR_ETAPA
            ),
            "siguiente_etapa": etapa_pendiente(meta),
            "registro_completo": (
                total >= TOTAL_FOTOS
            ),
        }
    )


@app.post("/api/registro/finalizar")
def finalizar():
    datos = request.get_json(silent=True) or {}
    sesion = str(datos.get("sesion", "")).strip()

    try:
        meta = leer_meta(sesion)
    except (ValueError, FileNotFoundError) as exc:
        return respuesta_error(str(exc), 404)

    incompletas = [
        etapa
        for etapa in ETAPAS
        if int(
            meta["conteos"].get(etapa, 0)
        ) != FOTOS_POR_ETAPA
    ]

    if incompletas:
        return respuesta_error(
            "Faltan capturas",
            etapas_incompletas=incompletas,
        )

    temporal = sesion_dir(sesion)
    archivos = sorted(
        (temporal / "embeddings").glob("*.npy")
    )

    if len(archivos) != TOTAL_FOTOS:
        return respuesta_error(
            (
                f"Se esperaban {TOTAL_FOTOS} "
                f"embeddings y existen {len(archivos)}"
            ),
            500,
        )

    embeddings = np.vstack(
        [
            np.load(ruta).reshape(1, -1)
            for ruta in archivos
        ]
    ).astype(np.float32)

    etapas = np.array(
        [
            ruta.stem.split("_")[1]
            for ruta in archivos
        ]
    )

    nombre = meta["nombre"]
    nombre_seguro = meta["nombre_seguro"]
    sesion_final = (
        "sesion_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    final = (
        DATOS_DIR
        / nombre_seguro
        / sesion_final
    )

    final.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal.rename(final)

    np.savez_compressed(
        final / "embeddings.npz",
        nombre=nombre,
        embeddings=embeddings,
        etapas=etapas,
    )

    shutil.rmtree(
        final / "embeddings",
        ignore_errors=True,
    )

    invalidar_galeria()

    try:
        ruta_remota = sincronizar_supabase(
            nombre,
            nombre_seguro,
            sesion_final,
            final,
            len(embeddings),
        )
    except Exception as exc:
        return respuesta_error(
            (
                "Se guardó localmente, pero falló Supabase: "
                f"{type(exc).__name__}: {exc}"
            ),
            502,
            guardado_local=True,
            carpeta=str(final),
        )

    return jsonify(
        {
            "ok": True,
            "mensaje": (
                "Registro guardado localmente "
                "y en Supabase"
            ),
            "nombre": nombre,
            "sesion": sesion_final,
            "cantidad_muestras": len(embeddings),
            "ruta_embeddings": ruta_remota,
        }
    )


@app.errorhandler(413)
def imagen_grande(_):
    return respuesta_error(
        "La imagen supera el tamaño permitido",
        413,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        threaded=True,
    )
