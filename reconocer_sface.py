import cv2
import numpy as np
import os
import time
from collections import deque


# =========================================================
# CONFIGURACIÓN
# =========================================================

MODELO_YUNET = os.path.join(
    "modelos",
    "face_detection_yunet_2023mar.onnx"
)

MODELO_SFACE = os.path.join(
    "modelos",
    "face_recognition_sface_2021dec.onnx"
)

CARPETA_DATOS = "datos_sface"

CONFIANZA_DETECCION = 0.70

# Más alto = menos falsos positivos.
UMBRAL_SIMILITUD = 0.52

# Evita aceptar cuando dos personas registradas dan resultados parecidos.
MARGEN_MINIMO = 0.05

TOP_MUESTRAS = 3
VOTOS_CONFIRMACION = 3
VENTANA_VOTOS = 7
FRAMES_MEMORIA = 15
FRAMES_ELIMINAR_TRACK = 8

TAMANO_MINIMO_CARA = 60
INDICES_CAMARA = [0, 1, 2, 3, 4]


# =========================================================
# UTILIDADES
# =========================================================

def normalizar_vector(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norma = np.linalg.norm(vector)

    if norma > 0:
        vector = vector / norma

    return vector


def normalizar_matriz(matriz):
    matriz = np.asarray(matriz, dtype=np.float32)

    if matriz.ndim == 1:
        matriz = matriz.reshape(1, -1)

    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0

    return matriz / normas


def dibujar_texto(frame, texto, x, y, color, escala=0.65):
    cv2.putText(
        frame,
        texto,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        texto,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color,
        2,
        cv2.LINE_AA
    )


def abrir_camara():
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("AUTOMATICO", cv2.CAP_ANY),
        ("MSMF", cv2.CAP_MSMF)
    ]

    for indice in INDICES_CAMARA:
        for nombre_backend, backend in backends:
            print(
                f"Probando cámara {indice} "
                f"con {nombre_backend}..."
            )

            cap = cv2.VideoCapture(indice, backend)

            if not cap.isOpened():
                cap.release()
                continue

            if backend == cv2.CAP_DSHOW:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG")
                )
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)

            for _ in range(20):
                try:
                    ret, frame = cap.read()
                except cv2.error:
                    ret = False
                    frame = None

                if ret and frame is not None and frame.size > 0:
                    alto, ancho = frame.shape[:2]

                    print(
                        f"Cámara encontrada: índice {indice}, "
                        f"{nombre_backend}, {ancho}x{alto}"
                    )

                    return cap

                time.sleep(0.05)

            cap.release()

    return None


# =========================================================
# BASE DE PERSONAS REGISTRADAS
# =========================================================

def cargar_base():
    if not os.path.isdir(CARPETA_DATOS):
        raise FileNotFoundError(
            "No existe datos_sface. "
            "Primero ejecuta registrar_sface.py."
        )

    acumulados = {}

    for raiz, _, archivos in os.walk(CARPETA_DATOS):
        if "embeddings.npz" not in archivos:
            continue

        ruta = os.path.join(raiz, "embeddings.npz")

        try:
            datos = np.load(ruta, allow_pickle=False)
            nombre = str(datos["nombre"].item()).strip()
            embeddings = normalizar_matriz(datos["embeddings"])

            if nombre and embeddings.size > 0:
                acumulados.setdefault(nombre, []).append(
                    embeddings
                )

        except Exception as error:
            print(f"No se pudo cargar {ruta}: {error}")

    base = {
        nombre: np.vstack(grupos).astype(np.float32)
        for nombre, grupos in acumulados.items()
    }

    if not base:
        raise RuntimeError(
            "No se encontraron registros completos."
        )

    print("\nPersonas cargadas:")

    for nombre, embeddings in base.items():
        print(f"- {nombre}: {len(embeddings)} muestras")

    print()

    return base


def identificar(embedding, base):
    resultados = []

    for nombre, embeddings in base.items():
        similitudes = embeddings @ embedding

        cantidad = min(
            TOP_MUESTRAS,
            len(similitudes)
        )

        mejores = np.sort(similitudes)[-cantidad:]
        puntuacion = float(np.mean(mejores))

        resultados.append(
            (nombre, puntuacion)
        )

    resultados.sort(
        key=lambda dato: dato[1],
        reverse=True
    )

    mejor_nombre, mejor_puntuacion = resultados[0]

    if len(resultados) > 1:
        segunda_puntuacion = resultados[1][1]
        margen = mejor_puntuacion - segunda_puntuacion
    else:
        margen = 1.0

    aceptado = (
        mejor_puntuacion >= UMBRAL_SIMILITUD
        and margen >= MARGEN_MINIMO
    )

    if aceptado:
        return mejor_nombre, mejor_puntuacion, margen

    return "Desconocido", mejor_puntuacion, margen


# =========================================================
# TRACKING: MEMORIA SEPARADA PARA CADA CARA
# =========================================================

def centro(bbox):
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def distancia_bboxes(bbox_a, bbox_b):
    ax, ay = centro(bbox_a)
    bx, by = centro(bbox_b)

    return float(np.hypot(ax - bx, ay - by))


def crear_track(track_id, deteccion):
    return {
        "id": track_id,
        "bbox": deteccion["bbox"],
        "feature": deteccion["feature"],
        "historial": deque(maxlen=VENTANA_VOTOS),
        "nombre_estable": None,
        "memoria": 0,
        "perdidos": 0,
        "nombre_actual": "Desconocido",
        "puntuacion": deteccion["puntuacion"],
        "margen": deteccion["margen"],
        "estado": "desconocido"
    }


def actualizar_track(track, deteccion):
    track["bbox"] = deteccion["bbox"]
    track["perdidos"] = 0
    track["nombre_actual"] = deteccion["nombre"]
    track["puntuacion"] = deteccion["puntuacion"]
    track["margen"] = deteccion["margen"]

    if deteccion["feature"] is not None:
        track["feature"] = deteccion["feature"]

    track["historial"].append(
        deteccion["nombre"]
    )

    conocidos = [
        nombre
        for nombre in track["historial"]
        if nombre != "Desconocido"
    ]

    if conocidos:
        conteos = {
            nombre: conocidos.count(nombre)
            for nombre in set(conocidos)
        }

        candidato = max(
            conteos,
            key=conteos.get
        )

        if conteos[candidato] >= VOTOS_CONFIRMACION:
            track["nombre_estable"] = candidato

    if (
        track["nombre_estable"] is not None
        and deteccion["nombre"] == track["nombre_estable"]
    ):
        track["memoria"] = FRAMES_MEMORIA
        track["estado"] = "reconocido"

    elif (
        track["nombre_estable"] is not None
        and track["memoria"] > 0
    ):
        track["memoria"] -= 1
        track["estado"] = "memoria"

    elif deteccion["nombre"] != "Desconocido":
        track["estado"] = "verificando"

    else:
        track["nombre_estable"] = None
        track["estado"] = "desconocido"


def asignar_tracks(detecciones, tracks, siguiente_id):
    disponibles = set(tracks.keys())
    asignaciones = []

    for indice, deteccion in enumerate(detecciones):
        mejor_id = None
        mejor_valor = float("inf")

        x1, y1, x2, y2 = deteccion["bbox"]
        tamano = max(x2 - x1, y2 - y1, 1)

        for track_id in disponibles:
            track = tracks[track_id]

            distancia = distancia_bboxes(
                deteccion["bbox"],
                track["bbox"]
            )

            distancia_maxima = max(
                120.0,
                tamano * 1.8
            )

            if distancia > distancia_maxima:
                continue

            similitud = 0.0

            if (
                deteccion["feature"] is not None
                and track["feature"] is not None
            ):
                similitud = float(
                    np.dot(
                        deteccion["feature"],
                        track["feature"]
                    )
                )

                if similitud < 0.18:
                    continue

            valor = distancia - similitud * 100.0

            if valor < mejor_valor:
                mejor_valor = valor
                mejor_id = track_id

        if mejor_id is None:
            mejor_id = siguiente_id
            siguiente_id += 1

            tracks[mejor_id] = crear_track(
                mejor_id,
                deteccion
            )

        else:
            disponibles.remove(mejor_id)

        asignaciones.append(
            (indice, mejor_id)
        )

    ids_asignados = {
        track_id
        for _, track_id in asignaciones
    }

    for track_id in list(tracks.keys()):
        if track_id not in ids_asignados:
            tracks[track_id]["perdidos"] += 1

            if (
                tracks[track_id]["perdidos"]
                > FRAMES_ELIMINAR_TRACK
            ):
                del tracks[track_id]

    for indice, track_id in asignaciones:
        actualizar_track(
            tracks[track_id],
            detecciones[indice]
        )

    return asignaciones, siguiente_id


def eliminar_nombres_duplicados(tracks, asignaciones):
    grupos = {}

    for _, track_id in asignaciones:
        nombre = tracks[track_id]["nombre_estable"]

        if nombre is not None:
            grupos.setdefault(nombre, []).append(
                track_id
            )

    for ids in grupos.values():
        if len(ids) <= 1:
            continue

        ganador = max(
            ids,
            key=lambda track_id: (
                tracks[track_id]["estado"] == "reconocido",
                tracks[track_id]["puntuacion"],
                tracks[track_id]["margen"]
            )
        )

        for track_id in ids:
            if track_id == ganador:
                continue

            tracks[track_id]["nombre_estable"] = None
            tracks[track_id]["memoria"] = 0
            tracks[track_id]["historial"].clear()
            tracks[track_id]["estado"] = "desconocido"


# =========================================================
# INICIAR MODELOS
# =========================================================

if not os.path.exists(MODELO_YUNET):
    raise FileNotFoundError(
        f"No se encontró: {MODELO_YUNET}"
    )

if not os.path.exists(MODELO_SFACE):
    raise FileNotFoundError(
        f"No se encontró: {MODELO_SFACE}"
    )

base = cargar_base()

detector = cv2.FaceDetectorYN.create(
    MODELO_YUNET,
    "",
    (320, 320),
    CONFIANZA_DETECCION,
    0.30,
    5000
)

reconocedor = cv2.FaceRecognizerSF.create(
    MODELO_SFACE,
    ""
)

cap = abrir_camara()

if cap is None:
    print("No se pudo abrir ninguna cámara.")
    raise SystemExit


# =========================================================
# RECONOCIMIENTO
# =========================================================

tracks = {}
siguiente_id = 1

print("Reconocimiento iniciado.")
print("Presiona Q para salir.")

while True:
    try:
        ret, frame = cap.read()
    except cv2.error as error:
        print(f"Error de cámara: {error}")
        break

    if not ret or frame is None or frame.size == 0:
        print("La cámara devolvió un frame vacío.")
        break

    alto, ancho = frame.shape[:2]
    detector.setInputSize((ancho, alto))

    try:
        _, caras = detector.detect(frame)
    except cv2.error as error:
        print(f"Error de YuNet: {error}")
        break

    detecciones = []

    if caras is not None:
        for cara in caras:
            x, y, w, h = cara[:4].astype(int)

            if (
                w < TAMANO_MINIMO_CARA
                or h < TAMANO_MINIMO_CARA
            ):
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(ancho - 1, x + w)
            y2 = min(alto - 1, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            embedding = None
            nombre = "Desconocido"
            puntuacion = 0.0
            margen = 0.0

            try:
                rostro = reconocedor.alignCrop(
                    frame,
                    cara
                )

                if rostro is not None and rostro.size > 0:
                    embedding = reconocedor.feature(
                        rostro
                    )

                    embedding = normalizar_vector(
                        embedding
                    )

                    (
                        nombre,
                        puntuacion,
                        margen
                    ) = identificar(
                        embedding,
                        base
                    )

            except cv2.error:
                pass

            detecciones.append({
                "bbox": (x1, y1, x2, y2),
                "feature": embedding,
                "nombre": nombre,
                "puntuacion": puntuacion,
                "margen": margen
            })

    asignaciones, siguiente_id = asignar_tracks(
        detecciones,
        tracks,
        siguiente_id
    )

    eliminar_nombres_duplicados(
        tracks,
        asignaciones
    )

    for indice, track_id in asignaciones:
        deteccion = detecciones[indice]
        track = tracks[track_id]

        x1, y1, x2, y2 = deteccion["bbox"]

        if track["estado"] == "reconocido":
            texto = track["nombre_estable"]
            color = (0, 255, 0)

        elif track["estado"] == "memoria":
            texto = track["nombre_estable"]
            color = (0, 255, 255)

        elif track["estado"] == "verificando":
            texto = "Verificando..."
            color = (0, 165, 255)

        else:
            texto = "Desconocido"
            color = (0, 0, 255)

        porcentaje = max(
            0.0,
            min(
                100.0,
                track["puntuacion"] * 100.0
            )
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        dibujar_texto(
            frame,
            f"{texto} | {porcentaje:.1f}%",
            x1,
            max(25, y1 - 12),
            color
        )

        dibujar_texto(
            frame,
            f"ID {track_id}",
            x1,
            min(alto - 10, y2 + 25),
            color,
            0.55
        )

    dibujar_texto(
        frame,
        f"Caras detectadas: {len(asignaciones)}",
        20,
        35,
        (255, 255, 255),
        0.70
    )

    cv2.imshow(
        "Reconocimiento YuNet + SFace",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
