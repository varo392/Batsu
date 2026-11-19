"""
config.py — Parámetros de diseño del simulador BATSÚ.

Cada constante referencia la tabla o sección del documento de tesis
(Proyecto BATSÚ v8) de donde proviene. Ajustar aquí, no en el código.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas — se calculan a partir de la ubicación de este archivo:
#   <BASE_DIR>/batsu/config.py  ->  BASE_DIR = carpeta raíz del proyecto
# Funciona igual en Colab (Drive montado) o en una máquina local.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos"
CORPUS_DIR = BASE_DIR / "corpus"

SEMILLAS_CSV = DATOS_DIR / "semillas_maestra.csv"
REACCION_CSV = DATOS_DIR / "tabla_reaccion.csv"
MODIFICADORES_CSV = DATOS_DIR / "modificadores_reaccion.csv"

# ---------------------------------------------------------------------------
# Categorías del vector de estado (Tabla 5.4) — el orden define el índice
# ---------------------------------------------------------------------------
SEGMENTOS = ["nacional", "internacional"]
TIPOS_TURISMO = ["aventura", "cultural", "naturaleza", "playa", "rural"]
ESTADOS_AFECTIVOS = ["ansioso", "desconfiado", "entusiasmado", "frustrado", "indeciso"]
ETAPAS_AIDA = ["atencion", "interes", "deseo", "accion"]
PRESUPUESTOS = ["bajo", "medio", "alto"]
DISTANCIA_CATEGORIAS = ["cercana", "intermedia", "lejana", "inviable"]
ACTIVOS = ["texto", "imagen", "video", "ubicacion", "paquete"]

# Codificación ordinal normalizada (Tabla 5.4)
AIDA_NORM = {"atencion": 0.0, "interes": 0.33, "deseo": 0.67, "accion": 1.0}
PRESUPUESTO_NORM = {"bajo": 0.0, "medio": 0.5, "alto": 1.0}
DISTANCIA_CAT_NORM = {"cercana": 0.0, "intermedia": 0.33, "lejana": 0.67, "inviable": 1.0}

# ---------------------------------------------------------------------------
# Espacio de acciones (Tabla 5.8)
# ---------------------------------------------------------------------------
ACCIONES = [
    "solicitar_informacion",   # 0
    "ofrecer_paquete_bajo",    # 1
    "ofrecer_paquete_medio",   # 2
    "ofrecer_paquete_alto",    # 3
    "enviar_activo",           # 4
    "escalar_humano",          # 5
    "cerrar_conversacion",     # 6
]
ACCIONES_OFERTA = {1, 2, 3}
ACCION_ENVIAR_ACTIVO = 4
ACCION_CERRAR = 6
PRESUPUESTO_DE_OFERTA = {1: "bajo", 2: "medio", 3: "alto"}
NIVEL_PRESUPUESTO = {"bajo": 0, "medio": 1, "alto": 2}

# Orden fijo del activo enviado en la acción 4 (Tabla 5.8, decisión de diseño)
ORDEN_ACTIVOS_ENVIO = ["ubicacion", "imagen", "video", "paquete"]

# ---------------------------------------------------------------------------
# Distancia y Trust Gate (Tabla 4.4, Sección 2.2)
# ---------------------------------------------------------------------------
UMBRAL_CERCANA_KM = 40.0
UMBRAL_INTERMEDIA_KM = 100.0
D_MAX_POR_OPERADOR = {"operador_local": 40, "operador_regional": 120, "agencia_con_transporte": 250}

# ---------------------------------------------------------------------------
# Recompensa (Tablas 5.9 y 5.10)
# ---------------------------------------------------------------------------
R_AVANCE_AIDA = 0.5
R_ESTANCAMIENTO = -0.1
R_VIOLACION_TRUST_GATE = -2.0
R_TERMINAL = {
    "conversion": 10.0,
    "avance_parcial": 2.0,
    "rechazo": -5.0,
    "abandono": -3.0,
    "recomendacion_inviable": -8.0,
    "finalizacion_limite_turnos": -1.0,
}
# Desenlaces que terminan el episodio (Tabla 5.11); avance_parcial NO termina
DESENLACES_TERMINALES = {"conversion", "rechazo", "abandono", "recomendacion_inviable"}

# ---------------------------------------------------------------------------
# Dinámica de transición (Sección 5.4.6, Tablas 5.11.a y 5.11.b)
# ---------------------------------------------------------------------------
P_CIERRE_TOPE = 0.95
PRIOR_PUNTOS = 20          # M8 (decisión D1)
TURNO_MODIFICADOR_M7 = 10  # M7: p(cierre) x 1.2 a partir de este turno
TURNO_LIMITE = 12          # M7: truncación
M1_P_CIERRE = 0.85
M2_FACTOR = 1.5
M2_TRANSFERENCIA = 25      # puntos conversión -> rechazo
M4_FACTOR = 0.7
M4_TRANSFERENCIA = 15      # puntos abandono -> conversión
M6_FACTOR = 1.4
M6_PUNTOS_CONVERSION = 20
M7_FACTOR = 1.2

# ---------------------------------------------------------------------------
# Generación del corpus (Sección 3.5, Sección 3.4)
# ---------------------------------------------------------------------------
CORRIDAS_POR_SEMILLA = 50
EPSILON_HEURISTICA = 0.20
SEMILLA_ALEATORIA_GLOBAL = 20260925

# Verificación del prior (decisión D1): fracción de semillas cuyo desenlace
# modal coincide con el resultado esperado
PRIOR_FIDELIDAD_MIN = 0.50
PRIOR_FIDELIDAD_MAX = 0.65
