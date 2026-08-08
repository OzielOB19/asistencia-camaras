const estado = document.getElementById("estado");
const instruccion = document.getElementById("instruccion");
const contador = document.getElementById("contador");
const progreso = document.getElementById("progreso");
const video = document.getElementById("camara");
const canvas = document.getElementById("canvas");

const etapas = [
    { nombre: "FRENTE", texto: "Mira al frente" },
    { nombre: "IZQUIERDA", texto: "Gira la cabeza a la izquierda" },
    { nombre: "DERECHA", texto: "Gira la cabeza a la derecha" },
    { nombre: "ARRIBA", texto: "Mira hacia arriba" },
    { nombre: "ABAJO", texto: "Mira hacia abajo" }
];

const fotosPorEtapa = 10;
const totalFotos = etapas.length * fotosPorEtapa;

let stream = null;
let capturaActiva = false;
let totalCapturadas = 0;
let etapaActual = 0;
let fotosEtapa = 0;

function actualizarUI(textoEstado) {
    estado.textContent = textoEstado;
    contador.textContent = `${totalCapturadas} / ${totalFotos} fotos`;
    progreso.value = totalCapturadas;
}

async function iniciarCamara() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        return true;
    } catch (error) {
        console.error(error);
        actualizarUI("❌ No se pudo abrir la cámara");
        return false;
    }
}

function apagarCamara() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
    }
}

function capturarImagen() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL("image/jpeg", 0.9);
}

async function enviarFrame(nombre, matricula) {
    const etapa = etapas[etapaActual];
    const imagen = capturarImagen();

    const respuesta = await fetch("/api/capturar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            nombre: nombre,
            matricula: matricula,
            etapa: etapa.nombre,
            numero: fotosEtapa + 1,
            imagen: imagen
        })
    });

    const data = await respuesta.json();

    if (data.ok) {
        fotosEtapa++;
        totalCapturadas++;
        actualizarUI(`✅ Foto guardada: ${etapa.nombre} ${fotosEtapa}/${fotosPorEtapa}`);
    } else {
        actualizarUI(`⚠️ ${data.mensaje}`);
        if (data.error) console.log(data.error);
    }

    if (fotosEtapa >= fotosPorEtapa) {
        etapaActual++;
        fotosEtapa = 0;
    }

    if (etapaActual >= etapas.length) {
        capturaActiva = false;
        instruccion.textContent = "Captura finalizada";
        actualizarUI("✅ Captura completada");
        apagarCamara();
        return;
    }

    instruccion.textContent = etapas[etapaActual].texto;
}

async function cicloCaptura(nombre, matricula) {
    while (capturaActiva) {
        await enviarFrame(nombre, matricula);
        await new Promise(resolve => setTimeout(resolve, 700));
    }
}

document.getElementById("iniciar").addEventListener("click", async () => {
    const nombre = document.getElementById("nombre").value.trim();
    const matricula = document.getElementById("matricula").value.trim();

    if (!nombre || !matricula) {
        alert("Completa nombre y matrícula");
        return;
    }

    await fetch("/api/limpiar", { method: "POST" });

    const okCamara = await iniciarCamara();
    if (!okCamara) return;

    capturaActiva = true;
    totalCapturadas = 0;
    etapaActual = 0;
    fotosEtapa = 0;

    progreso.max = totalFotos;
    progreso.value = 0;
    instruccion.textContent = etapas[0].texto;
    actualizarUI("🚀 Captura iniciada");

    cicloCaptura(nombre, matricula);
});

document.getElementById("detener").addEventListener("click", () => {
    capturaActiva = false;
    apagarCamara();
    actualizarUI("⛔ Captura detenida");
});
