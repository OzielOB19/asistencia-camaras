import cv2
import numpy as np
import os
import re
import time
import shutil
from datetime import datetime


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

ETAPAS = [
    "FRENTE",
    "IZQUIERDA",
    "DERECHA",
    "ARRIBA",
    "ABAJO"
]

FOTOS_POR_ETAPA = 10
INTERVALO_CAPTURA = 0.45

TAMANO_MINIMO_ROSTRO = 100
NITIDEZ_MINIMA = 45.0
CONFIANZA_DETECCION = 0.70

INDICES_CAMARA = [0, 1, 2, 3, 4]


# =========================================================
# FUNCIONES
# =========================================================

def limpiar_nombre(nombre):
    nombre = nombre.strip()

    nombre = re.sub(
        r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ_-]",
        "_",
        nombre
    )

    nombre = re.sub(r"_+", "_", nombre)

    return nombre.strip("_")


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
                f"con backend {nombre_backend}..."
            )

            cap = cv2.VideoCapture(indice, backend)

            if not cap.isOpened():
                cap.release()
                continue

            # DirectShow suele funcionar mejor con cámaras USB.
            if backend == cv2.CAP_DSHOW:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG")
                )

                cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    1280
                )

                cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    720
                )

                cap.set(
                    cv2.CAP_PROP_FPS,
                    30
                )

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            time.sleep(0.6)

            frame_valido = None

            for _ in range(20):
                try:
                    ret, frame_prueba = cap.read()

                except cv2.error:
                    ret = False
                    frame_prueba = None

                if (
                    ret
                    and frame_prueba is not None
                    and frame_prueba.size > 0
                ):
                    frame_valido = frame_prueba
                    break

                time.sleep(0.05)

            if frame_valido is not None:
                alto, ancho = frame_valido.shape[:2]

                print(
                    f"Cámara encontrada: índice {indice}, "
                    f"backend {nombre_backend}, "
                    f"resolución {ancho}x{alto}"
                )

                return cap

            cap.release()

    return None


def calcular_nitidez(imagen):
    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    nitidez = cv2.Laplacian(
        gris,
        cv2.CV_64F
    ).var()

    return float(nitidez)


def normalizar_embedding(embedding):
    vector = embedding.flatten().astype(
        np.float32
    )

    norma = np.linalg.norm(vector)

    if norma > 0:
        vector = vector / norma

    return vector


def dibujar_texto(
    frame,
    texto,
    posicion,
    color,
    escala=0.7
):
    x, y = posicion

    # Borde negro para que el texto sea visible.
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


def mostrar_instruccion_etapa(etapa):
    instrucciones = {
        "FRENTE": "Mira de frente a la cámara",
        "IZQUIERDA": "Gira un poco la cara a tu izquierda",
        "DERECHA": "Gira un poco la cara a tu derecha",
        "ARRIBA": "Levanta un poco la cara",
        "ABAJO": "Baja un poco la cara"
    }

    return instrucciones.get(
        etapa,
        "Coloca la cara como se indica"
    )


# =========================================================
# COMPROBAR MODELOS
# =========================================================

if not os.path.exists(MODELO_YUNET):
    raise FileNotFoundError(
        f"No se encontró el modelo YuNet: "
        f"{MODELO_YUNET}"
    )

if not os.path.exists(MODELO_SFACE):
    raise FileNotFoundError(
        f"No se encontró el modelo SFace: "
        f"{MODELO_SFACE}"
    )


# =========================================================
# PEDIR NOMBRE
# =========================================================

nombre_original = input(
    "Escribe el nombre de la persona: "
).strip()

nombre_carpeta = limpiar_nombre(
    nombre_original
)

if not nombre_carpeta:
    print("El nombre no es válido.")
    raise SystemExit


# =========================================================
# CREAR CARPETA DE SESIÓN
# =========================================================

fecha_sesion = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

carpeta_persona = os.path.join(
    CARPETA_DATOS,
    nombre_carpeta
)

carpeta_sesion = os.path.join(
    carpeta_persona,
    f"sesion_{fecha_sesion}"
)

carpeta_imagenes = os.path.join(
    carpeta_sesion,
    "imagenes"
)

os.makedirs(
    carpeta_imagenes,
    exist_ok=True
)


# =========================================================
# CARGAR YUNET Y SFACE
# =========================================================

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


# =========================================================
# ABRIR CÁMARA
# =========================================================

cap = abrir_camara()

if cap is None:
    shutil.rmtree(
        carpeta_sesion,
        ignore_errors=True
    )

    print(
        "No se pudo encontrar una cámara disponible."
    )

    print(
        "Cierra Cámara de Windows, Discord, OBS, "
        "Teams, Zoom o el navegador."
    )

    raise SystemExit


# =========================================================
# VARIABLES DE REGISTRO
# =========================================================

indice_etapa = 0
capturas_etapa = 0

embeddings = []
etiquetas_etapa = []

capturando = False
registro_completo = False

ultima_captura = 0.0

mensaje_estado = (
    "Presiona ESPACIO para comenzar"
)

color_estado = (0, 255, 255)

print()
print("Controles:")
print("ESPACIO = iniciar cada etapa")
print("Q = cancelar y salir")
print()


# =========================================================
# CICLO PRINCIPAL
# =========================================================

while True:
    try:
        ret, frame = cap.read()

    except cv2.error as error:
        print(
            f"Error al leer la cámara: {error}"
        )
        break

    if (
        not ret
        or frame is None
        or frame.size == 0
    ):
        print(
            "La cámara devolvió un fotograma vacío."
        )
        break

    alto, ancho = frame.shape[:2]

    detector.setInputSize(
        (ancho, alto)
    )

    try:
        _, caras = detector.detect(frame)

    except cv2.error as error:
        print(
            f"Error de YuNet: {error}"
        )
        break

    cantidad_caras = (
        0
        if caras is None
        else len(caras)
    )

    rostro_valido = False
    rostro_alineado = None
    nitidez = 0.0

    # =====================================================
    # DIBUJAR CARAS DETECTADAS
    # =====================================================

    if caras is not None:
        for cara in caras:
            x, y, w, h = cara[:4].astype(int)

            puntuacion = float(
                cara[14]
            )

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                ancho - 1,
                x + w
            )

            y2 = min(
                alto - 1,
                y + h
            )

            if cantidad_caras == 1:
                color_recuadro = (0, 255, 0)

            else:
                color_recuadro = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color_recuadro,
                2
            )

            dibujar_texto(
                frame,
                f"Cara {puntuacion:.2f}",
                (x1, max(25, y1 - 10)),
                color_recuadro,
                0.55
            )

        # Solo se permite una persona durante el registro.
        if cantidad_caras == 1:
            cara = caras[0]

            _, _, w, h = cara[:4].astype(int)

            if (
                w >= TAMANO_MINIMO_ROSTRO
                and h >= TAMANO_MINIMO_ROSTRO
            ):
                try:
                    rostro_alineado = (
                        reconocedor.alignCrop(
                            frame,
                            cara
                        )
                    )

                    if (
                        rostro_alineado is not None
                        and rostro_alineado.size > 0
                    ):
                        nitidez = calcular_nitidez(
                            rostro_alineado
                        )

                        if nitidez >= NITIDEZ_MINIMA:
                            rostro_valido = True

                except cv2.error:
                    rostro_valido = False


    # =====================================================
    # REGISTRO TERMINADO
    # =====================================================

    if indice_etapa >= len(ETAPAS):
        registro_completo = True

        dibujar_texto(
            frame,
            "REGISTRO TERMINADO",
            (20, 60),
            (0, 255, 0),
            1.0
        )

        dibujar_texto(
            frame,
            "Guardando datos...",
            (20, 105),
            (255, 255, 255),
            0.7
        )

        cv2.imshow(
            "Registro SFace",
            frame
        )

        cv2.waitKey(900)

        break


    # =====================================================
    # INFORMACIÓN EN PANTALLA
    # =====================================================

    etapa_actual = ETAPAS[
        indice_etapa
    ]

    dibujar_texto(
        frame,
        f"Persona: {nombre_original}",
        (20, 35),
        (255, 255, 255)
    )

    dibujar_texto(
        frame,
        f"Etapa: {etapa_actual}",
        (20, 70),
        (0, 255, 255)
    )

    dibujar_texto(
        frame,
        mostrar_instruccion_etapa(
            etapa_actual
        ),
        (20, 105),
        (255, 255, 255),
        0.62
    )

    dibujar_texto(
        frame,
        (
            f"Capturas: "
            f"{capturas_etapa}/"
            f"{FOTOS_POR_ETAPA}"
        ),
        (20, 140),
        (0, 255, 255)
    )

    dibujar_texto(
        frame,
        f"Nitidez: {nitidez:.1f}",
        (20, 175),
        (255, 255, 255),
        0.60
    )


    # =====================================================
    # ESTADO DEL ROSTRO
    # =====================================================

    if cantidad_caras == 0:
        mensaje_estado = (
            "No se detecta ninguna cara"
        )

        color_estado = (0, 0, 255)

    elif cantidad_caras > 1:
        mensaje_estado = (
            "Debe aparecer solamente una persona"
        )

        color_estado = (0, 0, 255)

    elif not rostro_valido:
        mensaje_estado = (
            "Acércate y mantén la cara quieta"
        )

        color_estado = (0, 165, 255)

    elif not capturando:
        mensaje_estado = (
            "Presiona ESPACIO para comenzar"
        )

        color_estado = (0, 255, 255)

    else:
        mensaje_estado = "Capturando..."
        color_estado = (0, 255, 0)

        tiempo_actual = time.time()

        if (
            tiempo_actual - ultima_captura
            >= INTERVALO_CAPTURA
        ):
            try:
                embedding = reconocedor.feature(
                    rostro_alineado
                )

                embedding = normalizar_embedding(
                    embedding
                )

            except cv2.error as error:
                print(
                    "No se pudo crear el embedding: "
                    f"{error}"
                )

                capturando = False
                continue

            embeddings.append(
                embedding
            )

            etiquetas_etapa.append(
                etapa_actual
            )

            numero_total = len(
                embeddings
            )

            nombre_archivo = (
                f"{numero_total:03d}_"
                f"{etapa_actual}_"
                f"{capturas_etapa + 1:02d}.jpg"
            )

            ruta_imagen = os.path.join(
                carpeta_imagenes,
                nombre_archivo
            )

            guardado = cv2.imwrite(
                ruta_imagen,
                rostro_alineado
            )

            if not guardado:
                print(
                    "No se pudo guardar: "
                    f"{ruta_imagen}"
                )

                embeddings.pop()
                etiquetas_etapa.pop()

                capturando = False
                continue

            capturas_etapa += 1
            ultima_captura = tiempo_actual

            print(
                f"{etapa_actual}: "
                f"{capturas_etapa}/"
                f"{FOTOS_POR_ETAPA}"
            )

            if (
                capturas_etapa
                >= FOTOS_POR_ETAPA
            ):
                indice_etapa += 1
                capturas_etapa = 0
                capturando = False

                if indice_etapa < len(ETAPAS):
                    siguiente = ETAPAS[
                        indice_etapa
                    ]

                    mensaje_estado = (
                        f"Prepárate para {siguiente} "
                        f"y presiona ESPACIO"
                    )

                    color_estado = (
                        0,
                        255,
                        255
                    )


    # =====================================================
    # MOSTRAR VENTANA
    # =====================================================

    dibujar_texto(
        frame,
        mensaje_estado,
        (20, alto - 30),
        color_estado,
        0.65
    )

    cv2.imshow(
        "Registro SFace",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF


    # =====================================================
    # CONTROLES
    # =====================================================

    if tecla == ord("q"):
        print("Registro cancelado.")
        break

    if tecla == 32:
        if cantidad_caras != 1:
            print(
                "Debe aparecer solamente una persona."
            )

        elif not rostro_valido:
            print(
                "La cara no tiene tamaño o "
                "nitidez suficiente."
            )

        else:
            capturando = True
            ultima_captura = 0.0

            print(
                f"Iniciando etapa: "
                f"{ETAPAS[indice_etapa]}"
            )


# =========================================================
# CERRAR CÁMARA
# =========================================================

cap.release()
cv2.destroyAllWindows()


# =========================================================
# GUARDAR RESULTADOS
# =========================================================

cantidad_esperada = (
    len(ETAPAS)
    * FOTOS_POR_ETAPA
)

if (
    registro_completo
    and len(embeddings) == cantidad_esperada
):
    embeddings_array = np.vstack(
        embeddings
    ).astype(np.float32)

    etapas_array = np.array(
        etiquetas_etapa
    )

    ruta_embeddings = os.path.join(
        carpeta_sesion,
        "embeddings.npz"
    )

    np.savez_compressed(
        ruta_embeddings,
        nombre=nombre_original,
        embeddings=embeddings_array,
        etapas=etapas_array
    )

    print()
    print("Registro guardado correctamente.")
    print(f"Persona: {nombre_original}")

    print(
        f"Embeddings guardados: "
        f"{len(embeddings)}"
    )

    print(
        f"Carpeta: {carpeta_sesion}"
    )

else:
    # Elimina registros incompletos para no contaminar
    # posteriormente el reconocimiento facial.
    shutil.rmtree(
        carpeta_sesion,
        ignore_errors=True
    )

    print()

    print(
        "El registro no se completó y no se "
        "guardaron datos incompletos."
    )