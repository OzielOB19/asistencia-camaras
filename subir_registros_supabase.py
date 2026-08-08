import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from supabase_config import supabase


load_dotenv()

CARPETA_DATOS = Path("datos_sface")

SUPABASE_BUCKET = os.getenv(
    "SUPABASE_BUCKET",
    "rostros"
).strip()

SUPABASE_TABLE = os.getenv(
    "SUPABASE_TABLE",
    "perfiles_sface"
).strip()


def limpiar_ruta(texto):
    texto = texto.strip()
    texto = re.sub(
        r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ_-]",
        "_",
        texto
    )
    texto = re.sub(r"_+", "_", texto)
    return texto.strip("_")


def obtener_datos_npz(ruta_npz):
    datos = np.load(
        ruta_npz,
        allow_pickle=False
    )

    nombre = str(
        datos["nombre"].item()
    ).strip()

    embeddings = datos["embeddings"]

    cantidad_muestras = int(
        len(embeddings)
    )

    return nombre, cantidad_muestras


def subir_embedding(ruta_npz):
    nombre, cantidad_muestras = obtener_datos_npz(
        ruta_npz
    )

    sesion = ruta_npz.parent.name

    nombre_ruta = limpiar_ruta(nombre)

    ruta_remota = (
        f"sface/{nombre_ruta}/"
        f"{sesion}/embeddings.npz"
    )

    print()
    print(f"Persona: {nombre}")
    print(f"Sesión: {sesion}")
    print(f"Muestras: {cantidad_muestras}")
    print(f"Destino: {ruta_remota}")

    with open(ruta_npz, "rb") as archivo:
        supabase.storage.from_(
            SUPABASE_BUCKET
        ).upload(
            path=ruta_remota,
            file=archivo,
            file_options={
                "content-type": "application/octet-stream",
                "upsert": "true"
            }
        )

    existente = (
        supabase
        .table(SUPABASE_TABLE)
        .select("id")
        .eq("ruta_embeddings", ruta_remota)
        .limit(1)
        .execute()
    )

    if existente.data:
        print("El registro ya existía en la tabla.")
        return

    respuesta = (
        supabase
        .table(SUPABASE_TABLE)
        .insert({
            "nombre": nombre,
            "sesion": sesion,
            "ruta_embeddings": ruta_remota,
            "cantidad_muestras": cantidad_muestras
        })
        .execute()
    )

    if not respuesta.data:
        raise RuntimeError(
            "Supabase no devolvió el registro insertado."
        )

    print("Registro agregado correctamente.")


def main():
    if not CARPETA_DATOS.exists():
        print(
            "No existe la carpeta datos_sface."
        )
        return

    archivos = list(
        CARPETA_DATOS.rglob(
            "embeddings.npz"
        )
    )

    if not archivos:
        print(
            "No se encontraron archivos embeddings.npz."
        )
        return

    print(
        f"Archivos encontrados: {len(archivos)}"
    )

    correctos = 0
    errores = 0

    for ruta_npz in archivos:
        try:
            subir_embedding(ruta_npz)
            correctos += 1

        except Exception as error:
            errores += 1

            print(
                f"ERROR con {ruta_npz}:"
            )
            print(type(error).__name__)
            print(error)

    print()
    print("==============================")
    print(f"Procesados correctamente: {correctos}")
    print(f"Errores: {errores}")
    print("==============================")


if __name__ == "__main__":
    main()
