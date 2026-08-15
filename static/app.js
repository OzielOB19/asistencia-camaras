const configRegistro = document.getElementById("configRegistro");

const cfg = JSON.parse(
    configRegistro?.textContent || "{}"
);

const video = document.getElementById("video");
const canvasCaptura = document.getElementById("canvasCaptura");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const nombre = document.getElementById("nombre");
const btnCamara = document.getElementById("btnCamara");
const btnReconocer = document.getElementById("btnReconocer");
const btnEtapa = document.getElementById("btnEtapa");

const mensaje = document.getElementById("mensaje");
const instruccion = document.getElementById("instruccion");
const estadoServidor = document.getElementById("estadoServidor");
const estadoPerfiles = document.getElementById("estadoPerfiles");
const resultadoReconocimiento = document.getElementById(
    "resultadoReconocimiento"
);

const etapaActual = document.getElementById("etapaActual");
const contador = document.getElementById("contador");
const total = document.getElementById("total");
const progreso = document.getElementById("progreso");

let stream = null;
let sesion = null;
let indiceEtapa = 0;
let capturandoRegistro = false;
let reconociendo = false;
let totalAceptadas = 0;

const CONFIG_RECONOCIMIENTO = {
    // YuNet rápido: mueve el cuadro. SFace: actualiza el nombre.
    anchoSeguimiento: 960,
    calidadSeguimiento: 0.72,
    periodoSeguimientoMs: 105,

    anchoIdentidad: 1440,
    calidadIdentidad: 0.86,
    periodoIdentidadMs: 1200,
    anchoIdentidadDetalle: 1920,
    calidadIdentidadDetalle: 0.90,
    detalleCada: 4,

    suavizadoCaja: 0.38,
    persistenciaMs: 520,
    conservarNombreMs: 3000,
    prediccionMaxMs: 85,
    maxVelocidadNorm: 1.6
};

let tracksRostros = [];
let siguienteTrackId = 1;
let animacionOverlay = null;
let controladorSeguimiento = null;
let controladorIdentidad = null;
let promesaSeguimiento = null;
let promesaIdentidad = null;
let contadorIdentidad = 0;
let ultimoFrameAnimacion = performance.now();
let metricaOverlay = {
    dpr: 1,
    escala: 1,
    offsetX: 0,
    offsetY: 0
};

const conteos = Object.fromEntries(
    cfg.etapas.map((etapa) => [etapa, 0])
);

const instrucciones = {
    FRENTE: "Mira de frente y mantén la cara quieta",
    IZQUIERDA: "Gira un poco hacia tu izquierda",
    DERECHA: "Gira un poco hacia tu derecha",
    ARRIBA: "Levanta un poco la cara",
    ABAJO: "Baja un poco la cara"
};

function etapa() {
    return cfg.etapas[indiceEtapa] ?? null;
}

function aviso(texto, tipo = "") {
    mensaje.textContent = texto;
    mensaje.className = `mensaje ${tipo}`;
}

function dormir(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function limpiarOverlay() {
    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );
}

function prepararOverlay() {
    if (!video.videoWidth || !video.videoHeight) {
        return;
    }

    const rect = video.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const anchoCanvas = Math.max(1, Math.round(rect.width * dpr));
    const altoCanvas = Math.max(1, Math.round(rect.height * dpr));

    if (
        overlay.width !== anchoCanvas
        || overlay.height !== altoCanvas
    ) {
        overlay.width = anchoCanvas;
        overlay.height = altoCanvas;
    }

    // El video usa object-fit: cover. Esta conversión hace que las
    // cajas sigan alineadas aunque el navegador recorte los bordes.
    const escala = Math.max(
        rect.width / video.videoWidth,
        rect.height / video.videoHeight
    );

    const anchoMostrado = video.videoWidth * escala;
    const altoMostrado = video.videoHeight * escala;

    metricaOverlay = {
        dpr,
        escala,
        offsetX: (rect.width - anchoMostrado) / 2,
        offsetY: (rect.height - altoMostrado) / 2
    };
}

function cajaEnPantalla(caja) {
    const { dpr, escala, offsetX, offsetY } = metricaOverlay;

    return {
        x: (
            offsetX
            + caja.x * video.videoWidth * escala
        ) * dpr,
        y: (
            offsetY
            + caja.y * video.videoHeight * escala
        ) * dpr,
        w: caja.w * video.videoWidth * escala * dpr,
        h: caja.h * video.videoHeight * escala * dpr
    };
}

function iouCajas(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);

    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const inter = interW * interH;
    const union = a.w * a.h + b.w * b.h - inter;

    return union > 0 ? inter / union : 0;
}

function distanciaCentros(a, b) {
    const ax = a.x + a.w / 2;
    const ay = a.y + a.h / 2;
    const bx = b.x + b.w / 2;
    const by = b.y + b.h / 2;
    return Math.hypot(ax - bx, ay - by);
}

function convertirCaja(caja, frameAncho, frameAlto) {
    return {
        x: caja.x / frameAncho,
        y: caja.y / frameAlto,
        w: caja.ancho / frameAncho,
        h: caja.alto / frameAlto
    };
}

function limitarVelocidad(valor) {
    const maximo = CONFIG_RECONOCIMIENTO.maxVelocidadNorm;
    return Math.max(-maximo, Math.min(maximo, valor));
}

function crearTrack(caja, ahora) {
    return {
        id: siguienteTrackId++,
        actual: { ...caja },
        objetivo: { ...caja },
        vx: 0,
        vy: 0,
        vw: 0,
        vh: 0,
        nombre: "desconocido",
        similitud: 0,
        ultimoNombreConocido: 0,
        ultimaVez: ahora,
        ultimaDeteccion: ahora
    };
}

function buscarTrackParaCaja(caja, usados = new Set()) {
    let mejor = null;
    let mejorPuntaje = -1;

    for (const track of tracksRostros) {
        if (usados.has(track.id)) continue;

        const iou = iouCajas(track.objetivo, caja);
        const distancia = distanciaCentros(track.objetivo, caja);
        const cercania = Math.max(0, 1 - distancia / 0.25);
        const puntaje = iou * 0.68 + cercania * 0.32;

        if (
            puntaje > mejorPuntaje
            && (iou > 0.015 || distancia < 0.13)
        ) {
            mejorPuntaje = puntaje;
            mejor = track;
        }
    }

    return mejor;
}

function actualizarTracksSeguimiento(datos) {
    const ahora = performance.now();
    const cajas = datos.rostros.map((rostro) =>
        convertirCaja(
            rostro.caja,
            datos.frame_ancho,
            datos.frame_alto
        )
    );

    const usados = new Set();

    for (const caja of cajas) {
        const track = buscarTrackParaCaja(caja, usados);

        if (track) {
            usados.add(track.id);
            const dt = Math.max(
                0.025,
                Math.min(0.35, (ahora - track.ultimaDeteccion) / 1000)
            );

            const centroAnteriorX = track.objetivo.x + track.objetivo.w / 2;
            const centroAnteriorY = track.objetivo.y + track.objetivo.h / 2;
            const centroNuevoX = caja.x + caja.w / 2;
            const centroNuevoY = caja.y + caja.h / 2;

            const vxMedido = (centroNuevoX - centroAnteriorX) / dt;
            const vyMedido = (centroNuevoY - centroAnteriorY) / dt;
            const vwMedido = (caja.w - track.objetivo.w) / dt;
            const vhMedido = (caja.h - track.objetivo.h) / dt;

            // Filtro de velocidad: reacciona rápido sin copiar el ruido de YuNet.
            track.vx = limitarVelocidad(track.vx * 0.55 + vxMedido * 0.45);
            track.vy = limitarVelocidad(track.vy * 0.55 + vyMedido * 0.45);
            track.vw = limitarVelocidad(track.vw * 0.65 + vwMedido * 0.35);
            track.vh = limitarVelocidad(track.vh * 0.65 + vhMedido * 0.35);

            track.objetivo = { ...caja };
            track.ultimaVez = ahora;
            track.ultimaDeteccion = ahora;
        } else {
            tracksRostros.push(crearTrack(caja, ahora));
        }
    }

    tracksRostros = tracksRostros.filter(
        (track) =>
            ahora - track.ultimaVez
            < CONFIG_RECONOCIMIENTO.persistenciaMs
    );

    actualizarTextoReconocimiento(ahora);
}

function aplicarIdentidades(datos) {
    const ahora = performance.now();
    const usados = new Set();

    for (const rostro of datos.rostros) {
        const caja = convertirCaja(
            rostro.caja,
            datos.frame_ancho,
            datos.frame_alto
        );

        let mejor = buscarTrackParaCaja(caja, usados);

        if (!mejor) {
            mejor = crearTrack(caja, ahora);
            tracksRostros.push(mejor);
        }

        usados.add(mejor.id);

        if (rostro.nombre !== "desconocido") {
            mejor.nombre = rostro.nombre;
            mejor.similitud = rostro.similitud;
            mejor.ultimoNombreConocido = ahora;
            mejor.fallosDesconocido = 0;
        } else {
            mejor.fallosDesconocido = (mejor.fallosDesconocido || 0) + 1;

            if (
                mejor.fallosDesconocido >= 3
                && ahora - mejor.ultimoNombreConocido
                    > CONFIG_RECONOCIMIENTO.conservarNombreMs
            ) {
                mejor.nombre = "desconocido";
                mejor.similitud = rostro.similitud;
            }
        }
    }

    // Si SFace no vio un track durante un rato, retirar el nombre viejo.
    for (const track of tracksRostros) {
        if (
            track.nombre !== "desconocido"
            && ahora - track.ultimoNombreConocido
                > CONFIG_RECONOCIMIENTO.conservarNombreMs
        ) {
            track.nombre = "desconocido";
            track.similitud = 0;
        }
    }

    actualizarTextoReconocimiento(ahora);
}

function actualizarTextoReconocimiento(ahora = performance.now()) {
    const nombres = tracksRostros
        .filter(
            (track) =>
                ahora - track.ultimaVez
                < CONFIG_RECONOCIMIENTO.persistenciaMs
        )
        .map((track) => {
            if (track.nombre === "desconocido") {
                return "desconocido";
            }

            const similitud = Number.isFinite(track.similitud)
                ? track.similitud.toFixed(2)
                : "0.00";

            return `${track.nombre} · ${similitud}`;
        });

    resultadoReconocimiento.textContent = nombres.length
        ? nombres.join(" | ")
        : "No se detectan rostros.";
}

function dibujarTrack(track, opacidad = 1) {
    const caja = cajaEnPantalla(track.actual);
    const x = caja.x;
    const y = caja.y;
    const w = caja.w;
    const h = caja.h;
    const conocido = track.nombre !== "desconocido";
    const dpr = metricaOverlay.dpr;

    const escalaFuenteCss = Math.max(
        16,
        Math.min(25, video.getBoundingClientRect().width * 0.023)
    );
    const escalaFuente = escalaFuenteCss * dpr;

    overlayCtx.save();
    overlayCtx.globalAlpha = opacidad;
    overlayCtx.lineWidth = 3 * dpr;
    overlayCtx.strokeStyle = conocido ? "#45df9c" : "#ff6363";
    overlayCtx.fillStyle = conocido
        ? "rgba(15, 111, 72, .92)"
        : "rgba(150, 36, 36, .92)";
    overlayCtx.font = `bold ${escalaFuente}px system-ui, sans-serif`;
    overlayCtx.textBaseline = "top";
    overlayCtx.lineJoin = "round";

    overlayCtx.strokeRect(x, y, w, h);

    const similitud = Number.isFinite(track.similitud)
        ? track.similitud.toFixed(2)
        : "0.00";

    const etiqueta = conocido
        ? `${track.nombre} · ${similitud}`
        : "desconocido";

    const padding = 8 * dpr;
    const altoEtiqueta = escalaFuente + padding * 1.35;
    const anchoEtiqueta = Math.min(
        overlay.width - Math.max(0, x),
        overlayCtx.measureText(etiqueta).width + padding * 2
    );
    const etiquetaY = Math.max(0, y - altoEtiqueta);

    overlayCtx.fillRect(
        x,
        etiquetaY,
        anchoEtiqueta,
        altoEtiqueta
    );

    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(
        etiqueta,
        x + padding,
        etiquetaY + padding * 0.45
    );
    overlayCtx.restore();
}

function animarSeguimiento(ahora) {
    if (!reconociendo) {
        animacionOverlay = null;
        return;
    }

    prepararOverlay();
    limpiarOverlay();

    const delta = Math.min(
        50,
        Math.max(1, ahora - ultimoFrameAnimacion)
    );
    ultimoFrameAnimacion = ahora;

    const base = CONFIG_RECONOCIMIENTO.suavizadoCaja;
    const factor = 1 - Math.pow(1 - base, delta / 16.67);

    for (const track of tracksRostros) {
        const edadDeteccion = ahora - track.ultimaDeteccion;
        const prediccionMs = Math.min(
            CONFIG_RECONOCIMIENTO.prediccionMaxMs,
            Math.max(0, edadDeteccion)
        );
        const t = prediccionMs / 1000;

        const objetivoPredicho = {
            x: track.objetivo.x + track.vx * t - track.vw * t / 2,
            y: track.objetivo.y + track.vy * t - track.vh * t / 2,
            w: track.objetivo.w + track.vw * t,
            h: track.objetivo.h + track.vh * t
        };

        objetivoPredicho.w = Math.max(0.01, objetivoPredicho.w);
        objetivoPredicho.h = Math.max(0.01, objetivoPredicho.h);
        objetivoPredicho.x = Math.max(
            -0.05,
            Math.min(1.05 - objetivoPredicho.w, objetivoPredicho.x)
        );
        objetivoPredicho.y = Math.max(
            -0.05,
            Math.min(1.05 - objetivoPredicho.h, objetivoPredicho.y)
        );

        for (const clave of ["x", "y", "w", "h"]) {
            track.actual[clave] += (
                objetivoPredicho[clave] - track.actual[clave]
            ) * factor;
        }

        const edad = ahora - track.ultimaVez;
        let opacidad = 1;

        if (edad > 320) {
            opacidad = Math.max(
                0,
                1 - (edad - 320)
                / (CONFIG_RECONOCIMIENTO.persistenciaMs - 320)
            );
        }

        if (opacidad > 0) {
            dibujarTrack(track, opacidad);
        }
    }

    tracksRostros = tracksRostros.filter(
        (track) =>
            ahora - track.ultimaVez
            < CONFIG_RECONOCIMIENTO.persistenciaMs
    );

    animacionOverlay = requestAnimationFrame(
        animarSeguimiento
    );
}

function iniciarAnimacionSeguimiento() {
    if (animacionOverlay !== null) {
        cancelAnimationFrame(animacionOverlay);
    }

    ultimoFrameAnimacion = performance.now();
    animacionOverlay = requestAnimationFrame(
        animarSeguimiento
    );
}

function actualizarRegistro() {
    const actual = etapa();

    etapaActual.textContent = actual ?? "Completado";

    contador.textContent = actual
        ? `${conteos[actual]} / ${cfg.fotosPorEtapa}`
        : `${cfg.fotosPorEtapa} / ${cfg.fotosPorEtapa}`;

    total.textContent =
        `${totalAceptadas} / ${cfg.totalFotos}`;

    progreso.style.width =
        `${(totalAceptadas / cfg.totalFotos) * 100}%`;

    document
        .querySelectorAll("#listaEtapas li")
        .forEach((elemento) => {
            const nombreEtapa = elemento.dataset.etapa;
            const cantidad = conteos[nombreEtapa];

            elemento.querySelector("small").textContent =
                `${cantidad} / ${cfg.fotosPorEtapa}`;

            elemento.classList.toggle(
                "activa",
                nombreEtapa === actual
                && cantidad < cfg.fotosPorEtapa
            );

            elemento.classList.toggle(
                "completa",
                cantidad >= cfg.fotosPorEtapa
            );
        });

    if (!reconociendo) {
        instruccion.textContent = actual
            ? instrucciones[actual]
            : "REGISTRO COMPLETADO";
    }
}

async function comprobarServidor() {
    try {
        const respuesta = await fetch(
            "/api/salud",
            { cache: "no-store" }
        );

        const datos = await respuesta.json();

        if (!respuesta.ok || !datos.ok) {
            throw new Error("Servidor no disponible");
        }

        estadoServidor.textContent = "Servidor conectado";
        estadoServidor.className = "estado conectado";

        estadoPerfiles.textContent =
            `${datos.personas_cargadas} personas · `
            + `${datos.muestras_cargadas} muestras`;

        estadoPerfiles.className = "estado neutro";
    } catch {
        estadoServidor.textContent = "Servidor sin conexión";
        estadoServidor.className = "estado error";
    }
}

async function abrirCamara() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1920 },
                height: { ideal: 1080 },
                frameRate: { ideal: 30, max: 30 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();
        prepararOverlay();

        const trackCamara = stream.getVideoTracks()[0];

        try {
            if (trackCamara && "contentHint" in trackCamara) {
                trackCamara.contentHint = "detail";
            }

            const capacidades = trackCamara?.getCapabilities?.() || {};
            const avanzadas = {};

            if (capacidades.focusMode?.includes?.("continuous")) {
                avanzadas.focusMode = "continuous";
            }
            if (capacidades.exposureMode?.includes?.("continuous")) {
                avanzadas.exposureMode = "continuous";
            }
            if (capacidades.whiteBalanceMode?.includes?.("continuous")) {
                avanzadas.whiteBalanceMode = "continuous";
            }

            if (Object.keys(avanzadas).length > 0) {
                await trackCamara.applyConstraints({
                    advanced: [avanzadas]
                });
            }
        } catch {
            // Algunas cámaras no exponen estos controles; no es un error.
        }

        const ajustes = trackCamara?.getSettings?.() || {};
        const resolucion = ajustes.width && ajustes.height
            ? `${ajustes.width}x${ajustes.height}`
            : `${video.videoWidth}x${video.videoHeight}`;

        btnCamara.textContent = "Cámara encendida";
        btnCamara.disabled = true;
        btnEtapa.disabled = false;
        btnReconocer.disabled = false;

        instruccion.textContent =
            "Elige registro o reconocimiento";

        aviso(`Cámara lista · ${resolucion}`, "ok");
    } catch {
        aviso(
            "La cámara requiere permiso y HTTPS o localhost.",
            "error"
        );
    }
}

function crearImagen(anchoMaximo = 1280, calidad = 0.90) {
    return new Promise((resolve, reject) => {
        if (!video.videoWidth || !video.videoHeight) {
            reject(
                new Error("La cámara todavía no está lista")
            );
            return;
        }

        const escala = Math.min(
            1,
            anchoMaximo / video.videoWidth
        );

        const ancho = Math.round(
            video.videoWidth * escala
        );

        const alto = Math.round(
            video.videoHeight * escala
        );

        canvasCaptura.width = ancho;
        canvasCaptura.height = alto;

        const ctx = canvasCaptura.getContext(
            "2d",
            { alpha: false }
        );

        ctx.save();
        ctx.translate(ancho, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, ancho, alto);
        ctx.restore();

        canvasCaptura.toBlob(
            (blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(
                        new Error(
                            "No se pudo crear la imagen"
                        )
                    );
                }
            },
            "image/jpeg",
            calidad
        );
    });
}

async function iniciarSesionRegistro() {
    const valor = nombre.value.trim();

    if (valor.length < 2) {
        aviso("Escribe un nombre válido.", "error");
        return false;
    }

    const respuesta = await fetch(
        "/api/registro/iniciar",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ nombre: valor })
        }
    );

    const datos = await respuesta.json();

    if (!respuesta.ok || !datos.ok) {
        aviso(
            datos.mensaje || "No se pudo iniciar.",
            "error"
        );
        return false;
    }

    sesion = datos.sesion;
    nombre.disabled = true;
    actualizarRegistro();
    return true;
}

async function enviarCapturaRegistro() {
    const actual = etapa();
    const imagen = await crearImagen(1280, 0.90);
    const formulario = new FormData();

    formulario.append("sesion", sesion);
    formulario.append("etapa", actual);
    formulario.append("image", imagen, "captura.jpg");

    const respuesta = await fetch(
        "/api/registro/capturar",
        {
            method: "POST",
            body: formulario
        }
    );

    const datos = await respuesta.json();

    if (!respuesta.ok || !datos.ok) {
        aviso(
            datos.mensaje || "Captura rechazada.",
            "aviso"
        );
        return datos;
    }

    conteos[actual] = datos.conteo;
    totalAceptadas = datos.total;
    actualizarRegistro();

    aviso(
        `Captura ${datos.conteo}/`
        + `${cfg.fotosPorEtapa} aceptada. `
        + `Nitidez: ${datos.nitidez}`,
        "ok"
    );

    return datos;
}

async function cicloRegistro() {
    capturandoRegistro = true;
    btnEtapa.disabled = true;
    btnReconocer.disabled = true;

    while (capturandoRegistro) {
        const actual = etapa();

        if (!actual) {
            break;
        }

        try {
            const datos = await enviarCapturaRegistro();

            if (datos?.registro_completo) {
                capturandoRegistro = false;
                await finalizarRegistro();
                return;
            }

            if (datos?.etapa_completa) {
                capturandoRegistro = false;
                indiceEtapa += 1;
                actualizarRegistro();

                const siguiente = etapa();

                if (siguiente) {
                    btnEtapa.textContent =
                        `Iniciar etapa ${siguiente}`;

                    btnEtapa.disabled = false;
                    btnReconocer.disabled = false;

                    aviso(
                        `${actual} completa. `
                        + `Prepárate para ${siguiente}.`,
                        "ok"
                    );
                }

                return;
            }
        } catch (error) {
            aviso(error.message, "error");
        }

        await dormir(650);
    }
}

async function finalizarRegistro() {
    btnEtapa.disabled = true;
    btnReconocer.disabled = true;

    instruccion.textContent =
        "Guardando localmente y en Supabase…";

    aviso(
        "Subiendo embeddings y fotografías…",
        "aviso"
    );

    const respuesta = await fetch(
        "/api/registro/finalizar",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ sesion })
        }
    );

    const datos = await respuesta.json();

    if (!respuesta.ok || !datos.ok) {
        aviso(
            datos.mensaje || "No se pudo finalizar.",
            "error"
        );

        btnReconocer.disabled = false;
        return;
    }

    actualizarRegistro();
    btnEtapa.textContent = "Registro terminado";
    btnReconocer.disabled = false;

    aviso(
        `${datos.nombre} quedó registrado con `
        + `${datos.cantidad_muestras} muestras.`,
        "ok"
    );

    await comprobarServidor();
}

async function enviarSeguimiento() {
    const imagen = await crearImagen(
        CONFIG_RECONOCIMIENTO.anchoSeguimiento,
        CONFIG_RECONOCIMIENTO.calidadSeguimiento
    );
    const formulario = new FormData();

    formulario.append(
        "image",
        imagen,
        "seguimiento.jpg"
    );

    controladorSeguimiento = new AbortController();

    const respuesta = await fetch(
        "/api/detectar",
        {
            method: "POST",
            body: formulario,
            signal: controladorSeguimiento.signal,
            cache: "no-store"
        }
    );

    const datos = await respuesta.json();

    if (!respuesta.ok || !datos.ok) {
        throw new Error(
            datos.mensaje || "Falló el seguimiento"
        );
    }

    actualizarTracksSeguimiento(datos);
    return datos;
}

async function cicloSeguimiento() {
    while (reconociendo) {
        const inicio = performance.now();

        try {
            await enviarSeguimiento();
        } catch (error) {
            if (
                error.name !== "AbortError"
                && reconociendo
            ) {
                aviso(error.message, "error");
            }
        } finally {
            controladorSeguimiento = null;
        }

        const usado = performance.now() - inicio;
        const espera = Math.max(
            12,
            CONFIG_RECONOCIMIENTO.periodoSeguimientoMs - usado
        );

        if (reconociendo) {
            await dormir(espera);
        }
    }
}

async function enviarIdentidad() {
    contadorIdentidad += 1;

    const modoDetalle = (
        contadorIdentidad
        % CONFIG_RECONOCIMIENTO.detalleCada === 0
    );

    const ancho = modoDetalle
        ? CONFIG_RECONOCIMIENTO.anchoIdentidadDetalle
        : CONFIG_RECONOCIMIENTO.anchoIdentidad;

    const calidad = modoDetalle
        ? CONFIG_RECONOCIMIENTO.calidadIdentidadDetalle
        : CONFIG_RECONOCIMIENTO.calidadIdentidad;

    const imagen = await crearImagen(ancho, calidad);
    const formulario = new FormData();

    formulario.append(
        "image",
        imagen,
        "identidad.jpg"
    );
    formulario.append(
        "detalle",
        modoDetalle ? "1" : "0"
    );

    controladorIdentidad = new AbortController();

    const respuesta = await fetch(
        "/api/reconocer",
        {
            method: "POST",
            body: formulario,
            signal: controladorIdentidad.signal,
            cache: "no-store"
        }
    );

    const datos = await respuesta.json();

    if (!respuesta.ok || !datos.ok) {
        throw new Error(
            datos.mensaje || "Falló la identidad"
        );
    }

    aplicarIdentidades(datos);
    return datos;
}

async function cicloIdentidad() {
    // Un pequeño desfase evita que YuNet rápido y SFace arranquen
    // exactamente al mismo instante.
    await dormir(120);

    while (reconociendo) {
        const inicio = performance.now();

        try {
            const datos = await enviarIdentidad();

            if (datos?.procesamiento_ms) {
                aviso(
                    `Seguimiento activo · identidad ${Math.round(datos.procesamiento_ms)} ms`,
                    "ok"
                );
            }
        } catch (error) {
            if (
                error.name !== "AbortError"
                && reconociendo
            ) {
                aviso(error.message, "error");
            }
        } finally {
            controladorIdentidad = null;
        }

        const usado = performance.now() - inicio;
        const espera = Math.max(
            40,
            CONFIG_RECONOCIMIENTO.periodoIdentidadMs - usado
        );

        if (reconociendo) {
            await dormir(espera);
        }
    }
}

function cicloReconocimiento() {
    reconociendo = true;
    contadorIdentidad = 0;
    tracksRostros = [];

    btnReconocer.textContent =
        "Detener reconocimiento";

    btnReconocer.classList.add("detener");
    btnEtapa.disabled = true;
    nombre.disabled = true;

    instruccion.textContent =
        "RECONOCIMIENTO ACTIVO";

    aviso(
        "Seguimiento rápido + identificación SFace activos…",
        "ok"
    );

    iniciarAnimacionSeguimiento();

    promesaSeguimiento = cicloSeguimiento();
    promesaIdentidad = cicloIdentidad();
}

function detenerReconocimiento() {
    reconociendo = false;

    controladorSeguimiento?.abort();
    controladorIdentidad?.abort();
    controladorSeguimiento = null;
    controladorIdentidad = null;
    promesaSeguimiento = null;
    promesaIdentidad = null;

    if (animacionOverlay !== null) {
        cancelAnimationFrame(animacionOverlay);
        animacionOverlay = null;
    }

    tracksRostros = [];
    limpiarOverlay();

    btnReconocer.textContent =
        "Iniciar reconocimiento";

    btnReconocer.classList.remove("detener");
    btnEtapa.disabled = false;

    if (!sesion) {
        nombre.disabled = false;
    }

    resultadoReconocimiento.textContent =
        "Reconocimiento detenido.";

    actualizarRegistro();
    aviso("Reconocimiento detenido.", "");
}

btnCamara.addEventListener(
    "click",
    abrirCamara
);

btnReconocer.addEventListener(
    "click",
    () => {
        if (!stream) {
            aviso(
                "Primero enciende la cámara.",
                "error"
            );
            return;
        }

        if (reconociendo) {
            detenerReconocimiento();
        } else {
            cicloReconocimiento();
        }
    }
);

btnEtapa.addEventListener(
    "click",
    async () => {
        if (!stream) {
            aviso(
                "Primero enciende la cámara.",
                "error"
            );
            return;
        }

        if (reconociendo) {
            detenerReconocimiento();
        }

        if (!sesion) {
            const correcto =
                await iniciarSesionRegistro();

            if (!correcto) {
                return;
            }
        }

        const actual = etapa();

        if (!actual) {
            return;
        }

        btnEtapa.textContent =
            `Capturando ${actual}…`;

        aviso(
            `Mantén la posición ${actual}.`,
            "aviso"
        );

        cicloRegistro();
    }
);

window.addEventListener("resize", prepararOverlay);
video.addEventListener("loadedmetadata", prepararOverlay);

window.addEventListener(
    "beforeunload",
    () => {
        controladorSeguimiento?.abort();
        controladorIdentidad?.abort();
        if (animacionOverlay !== null) {
            cancelAnimationFrame(animacionOverlay);
        }
        stream?.getTracks().forEach(
            (track) => track.stop()
        );
    }
);
const btnAsistencia = document.getElementById("btnAsistencia");
const btnCerrarAsistencia = document.getElementById("btnCerrarAsistencia");
const seccionAsistencia = document.getElementById("seccionAsistencia");

btnAsistencia?.addEventListener("click", () => {
    seccionAsistencia.hidden = false;

    seccionAsistencia.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
});

btnCerrarAsistencia?.addEventListener("click", () => {
    seccionAsistencia.hidden = true;
});
const fechaAsistencia = document.getElementById("fechaAsistencia");
const periodoActual = document.getElementById("periodoActual");
const totalAsistenciasHoy = document.getElementById("totalAsistenciasHoy");
const totalPersonasAsistencia = document.getElementById("totalPersonasAsistencia");

const listaHorasAsistencia = document.getElementById("listaHorasAsistencia");
const listaPersonasAsistencia = document.getElementById("listaPersonasAsistencia");

const btnVistaHoras = document.getElementById("btnVistaHoras");
const btnVistaPersonas = document.getElementById("btnVistaPersonas");

const vistaHorasAsistencia = document.getElementById("vistaHorasAsistencia");
const vistaPersonasAsistencia = document.getElementById("vistaPersonasAsistencia");


function escaparHTML(valor) {
    return String(valor ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


async function cargarAsistencias() {
    listaHorasAsistencia.textContent = "Cargando...";
    listaPersonasAsistencia.textContent = "Cargando...";

    try {
        const respuesta = await fetch("/api/asistencias/hoy", {
            cache: "no-store"
        });

        const datos = await respuesta.json();

        if (!respuesta.ok || !datos.ok) {
            throw new Error(
                datos.mensaje || "No se pudieron cargar las asistencias"
            );
        }

        fechaAsistencia.textContent =
            `${datos.dia} · ${datos.fecha}`;

        periodoActual.textContent =
            datos.periodo_actual;

        totalAsistenciasHoy.textContent =
            datos.total_asistencias;

        totalPersonasAsistencia.textContent =
            datos.total_personas;


        // =========================
        // VISTA POR HORAS
        // =========================

       // =========================
// VISTA POR HORAS - CALENDARIO
// =========================

const crearTarjetaHora = (hora) => {
    const personas = hora.personas || [];

    const esActual =
        datos.periodo_actual === hora.periodo;

    const lista = personas.length
        ? personas.map(persona => `
            <div class="calendario-persona">
                <div>
                    <span class="calendario-avatar">
                        ${escaparHTML(
                            String(persona.nombre || "?")
                                .charAt(0)
                                .toUpperCase()
                        )}
                    </span>

                    <strong>
                        ${escaparHTML(persona.nombre)}
                    </strong>
                </div>

                <time>
                    ${escaparHTML(
                        String(
                            persona.hora_deteccion || ""
                        ).slice(0, 5)
                    )}
                </time>
            </div>
        `).join("")
        : `
            <div class="calendario-sin-registros">
                Sin asistencias
            </div>
        `;

    return `
        <article
            class="calendario-hora
            ${esActual ? "calendario-hora-actual" : ""}"
        >
            <div class="calendario-hora-cabecera">
                <div>
                    <span class="calendario-periodo">
                        ${escaparHTML(hora.periodo)}
                    </span>

                    <strong class="calendario-hora-inicio">
                        ${escaparHTML(hora.hora_clase)}
                    </strong>
                </div>

                <span class="calendario-contador">
                    ${hora.asistieron}
                    ${hora.asistieron === 1
                        ? "persona"
                        : "personas"}
                </span>
            </div>

            <div class="calendario-hora-contenido">
                ${lista}
            </div>
        </article>
    `;
};


const horas = datos.horas || [];

const primeraParte = horas.slice(0, 3);
const segundaParte = horas.slice(3);

listaHorasAsistencia.innerHTML = `
    <div class="calendario-asistencia">

        <div class="calendario-fila calendario-fila-3">
            ${primeraParte.map(crearTarjetaHora).join("")}
        </div>

        <div class="calendario-descanso">
            <div class="calendario-descanso-icono">
                ☕
            </div>

            <div>
                <strong>DESCANSO</strong>
                <span>10:15 AM — 10:34 AM</span>
            </div>
        </div>

        <div class="calendario-fila calendario-fila-3">
            ${segundaParte
                .slice(0, 3)
                .map(crearTarjetaHora)
                .join("")}
        </div>

        <div class="calendario-fila calendario-fila-3">
            ${segundaParte
                .slice(3, 6)
                .map(crearTarjetaHora)
                .join("")}
        </div>

    </div>
`;  

        // =========================
        // VISTA POR PERSONA
        // =========================

        const personas = Object.entries(
            datos.personas || {}
        );

        if (!personas.length) {
            listaPersonasAsistencia.innerHTML =
                `<div class="asistencia-vacia">
                    No hay asistencias registradas hoy.
                </div>`;

            return;
        }

        listaPersonasAsistencia.innerHTML =
            personas.map(([nombre, registros]) => {

                const horas = registros.map(registro => `
                    <div class="asistencia-persona-hora">
                        <span>
                            ${escaparHTML(registro.periodo)}
                        </span>

                        <strong>
                            ${escaparHTML(registro.hora_deteccion)}
                        </strong>

                        <em>
                            ${escaparHTML(registro.estado)}
                        </em>
                    </div>
                `).join("");

                return `
                    <details class="asistencia-hora">
                        <summary>
                            <strong>
                                ${escaparHTML(nombre)}
                            </strong>

                            <b>
                                ${registros.length} / 9 horas
                            </b>
                        </summary>

                        <div class="asistencia-lista-personas">
                            ${horas}
                        </div>
                    </details>
                `;
            }).join("");

    } catch (error) {
        console.error(error);

        listaHorasAsistencia.textContent =
            "Error al cargar las asistencias.";

        listaPersonasAsistencia.textContent =
            "Error al cargar las asistencias.";
    }
}


btnAsistencia?.addEventListener("click", () => {
    cargarAsistencias();
});


btnVistaHoras?.addEventListener("click", () => {
    vistaHorasAsistencia.hidden = false;
    vistaPersonasAsistencia.hidden = true;

    btnVistaHoras.classList.add("principal");
    btnVistaPersonas.classList.remove("principal");
});


btnVistaPersonas?.addEventListener("click", () => {
    vistaHorasAsistencia.hidden = true;
    vistaPersonasAsistencia.hidden = false;

    btnVistaPersonas.classList.add("principal");
    btnVistaHoras.classList.remove("principal");
});

actualizarRegistro();
comprobarServidor();
