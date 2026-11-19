"""batsu_sim.py — Simulador BATSÚ (generado desde notebooks/01_simulador.ipynb; no editar a mano)."""
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / 'datos'
CORPUS_DIR = BASE_DIR / 'corpus'
# --- Categorías del vector de estado (Tabla 5.4) — el orden define el índice one-hot
SEGMENTOS = ['nacional', 'internacional']
TIPOS_TURISMO = ['aventura', 'cultural', 'naturaleza', 'playa', 'rural']
ESTADOS_AFECTIVOS = ['ansioso', 'desconfiado', 'entusiasmado', 'frustrado', 'indeciso']
ETAPAS_AIDA = ['atencion', 'interes', 'deseo', 'accion']
PRESUPUESTOS = ['bajo', 'medio', 'alto']
DISTANCIA_CATEGORIAS = ['cercana', 'intermedia', 'lejana', 'inviable']
ACTIVOS = ['texto', 'imagen', 'video', 'ubicacion', 'paquete']

AIDA_NORM = {'atencion': 0.0, 'interes': 0.33, 'deseo': 0.67, 'accion': 1.0}
PRESUPUESTO_NORM = {'bajo': 0.0, 'medio': 0.5, 'alto': 1.0}
DISTANCIA_CAT_NORM = {'cercana': 0.0, 'intermedia': 0.33, 'lejana': 0.67, 'inviable': 1.0}

# --- Espacio de acciones (Tabla 5.8)
ACCIONES = ['solicitar_informacion',   # 0
            'ofrecer_paquete_bajo',    # 1
            'ofrecer_paquete_medio',   # 2
            'ofrecer_paquete_alto',    # 3
            'enviar_activo',           # 4
            'escalar_humano',          # 5
            'cerrar_conversacion']     # 6
ACCIONES_OFERTA = {1, 2, 3}
ACCION_ENVIAR_ACTIVO = 4
ACCION_CERRAR = 6
PRESUPUESTO_DE_OFERTA = {1: 'bajo', 2: 'medio', 3: 'alto'}
NIVEL_PRESUPUESTO = {'bajo': 0, 'medio': 1, 'alto': 2}
ORDEN_ACTIVOS_ENVIO = ['ubicacion', 'imagen', 'video', 'paquete']   # regla de la acción 4

# --- Distancia y Trust Gate (Tabla 4.4, Sección 2.2)
UMBRAL_CERCANA_KM = 40.0
UMBRAL_INTERMEDIA_KM = 100.0
D_MAX_POR_OPERADOR = {'operador_local': 40, 'operador_regional': 120, 'agencia_con_transporte': 250}

# --- Recompensa (Tablas 5.9 y 5.10)
R_AVANCE_AIDA = 0.5
R_ESTANCAMIENTO = -0.1
R_VIOLACION_TRUST_GATE = -2.0
R_TERMINAL = {'conversion': 10.0, 'avance_parcial': 2.0, 'rechazo': -5.0, 'abandono': -3.0,
              'recomendacion_inviable': -8.0, 'finalizacion_limite_turnos': -1.0}
DESENLACES_TERMINALES = {'conversion', 'rechazo', 'abandono', 'recomendacion_inviable'}   # Tabla 5.11

# --- Dinámica de transición (Sección 5.4.6, Tablas 5.11.a y 5.11.b)
P_CIERRE_TOPE = 0.95
PRIOR_PUNTOS = 40            # M8, decisión D1; 20 fue el valor inicial (fidelidad 43 %), 40 da 60 % (Sección 5.4.6)
TURNO_MODIFICADOR_M7 = 10    # M7: p(cierre) x 1.2 desde este turno
TURNO_LIMITE = 12            # M7: truncación por límite de turnos
M1_P_CIERRE = 0.85
M2_FACTOR, M2_TRANSFERENCIA = 1.5, 25     # conversión -> rechazo
M4_FACTOR, M4_TRANSFERENCIA = 0.7, 15     # abandono -> conversión
M6_FACTOR, M6_PUNTOS_CONVERSION = 1.4, 20
M7_FACTOR = 1.2

# --- Generación del corpus (Secciones 3.4 y 3.5)
CORRIDAS_POR_SEMILLA = 100   # 50 en la estimación inicial; 100 para conservar ~26 000 transiciones con escala 0.5 (Sección 3.5)
EPSILON_HEURISTICA = 0.20
SEMILLA_ALEATORIA_GLOBAL = 20260925
PRIOR_FIDELIDAD_MIN, PRIOR_FIDELIDAD_MAX = 0.50, 0.65   # verificación de D1

##################### ---- revisar #############################

ESCALA_P_CIERRE = 0.5   # factor sobre p(cierre) de la celda base; 1.0 = tabla original (Sección 5.4.6)

print('Parámetros cargados. Acciones:', len(ACCIONES), '| Estados afectivos:', len(ESTADOS_AFECTIVOS))


# ---------- Semillas (Tabla 4.6) ----------
semillas = pd.read_csv(DATOS_DIR / 'semillas_maestra.csv')
semillas['activos_lista'] = semillas['activos_disponibles'].str.split('|')
semillas['inviable'] = semillas['inviable'].astype(str).str.lower().eq('true')

assert len(semillas) == 60, f'Se esperaban 60 semillas, hay {len(semillas)}'
assert semillas['id'].is_unique

def _chk(serie, esperado, nombre):
    obs = serie.value_counts().to_dict()
    assert obs == esperado, f'{nombre}: esperado {esperado}, observado {obs}'

_chk(semillas['segmento'], {'nacional': 30, 'internacional': 30}, 'segmento')
_chk(semillas['tipo_turismo'], {'playa': 15, 'naturaleza': 15, 'aventura': 10, 'cultural': 10, 'rural': 10}, 'tipo_turismo')
_chk(semillas['estado_afectivo_inicial'], {e: 12 for e in ESTADOS_AFECTIVOS}, 'estado_afectivo_inicial')
_chk(semillas['resultado_esperado'], {'conversion': 20, 'avance_parcial': 20, 'rechazo': 5, 'abandono': 5,
                                      'recomendacion_inviable': 5, 'finalizacion_limite_turnos': 5}, 'resultado_esperado')
_chk(semillas['distancia_categoria_calc'], {'cercana': 16, 'intermedia': 23, 'lejana': 16, 'inviable': 5}, 'distancia_categoria_calc')

# Los inviables son exactamente los episodios de recomendación inviable, y lo son por radio de cobertura
inv = semillas[semillas['inviable']]
assert set(inv['resultado_esperado']) == {'recomendacion_inviable'} and len(inv) == 5
assert (inv['distancia_haversine_km'] > inv['d_max_km']).all(), 'Algún inviable no excede su D_max'

# Verificación independiente de las bandas contra los umbrales de la Tabla 4.4
def banda(km):
    return 'cercana' if km < UMBRAL_CERCANA_KM else ('intermedia' if km <= UMBRAL_INTERMEDIA_KM else 'lejana')
calc = semillas.apply(lambda r: 'inviable' if r['inviable'] else banda(r['distancia_haversine_km']), axis=1)
assert (calc == semillas['distancia_categoria_calc']).all(), 'Las bandas del archivo no coinciden con los umbrales'

DIST_MAX_KM = semillas['distancia_haversine_km'].max()
print(f'Semillas: {len(semillas)} OK · distancia máxima {DIST_MAX_KM:.1f} km · inviables: {inv["id"].tolist()}')


# ---------- Tabla de reacción (Tabla 5.11.a) ----------
reaccion = pd.read_csv(DATOS_DIR / 'tabla_reaccion.csv')
PESOS = ['w_conversion', 'w_avance_parcial', 'w_rechazo', 'w_abandono']

assert len(reaccion) == 35, f'Se esperaban 35 celdas, hay {len(reaccion)}'
assert set(reaccion['estado_afectivo']) == set(ESTADOS_AFECTIVOS)
assert set(reaccion['accion_idx']) == set(range(7))
assert not reaccion.duplicated(['estado_afectivo', 'accion_idx']).any(), 'Celdas repetidas'
assert (reaccion[PESOS].sum(axis=1) == 100).all(), 'Hay celdas cuyos pesos no suman 100'
assert reaccion['p_cierre'].between(0, 1).all()

# Acción esperada por estado, según la columna de ajuste esperado de la Tabla 4.2
ESPERADAS_DOC = {'entusiasmado': {1, 2, 3}, 'indeciso': {4}, 'ansioso': {0}, 'desconfiado': {4}, 'frustrado': {0}}
for estado, esperadas in ESPERADAS_DOC.items():
    obs = set(reaccion[(reaccion['estado_afectivo'] == estado) & (reaccion['es_accion_esperada'] == 'si')]['accion_idx'])
    assert obs == esperadas, f'{estado}: acción esperada {obs}, documento {esperadas}'

# Índice (estado, acción) -> fila, para consulta O(1) en el simulador
REACCION = {(r.estado_afectivo, int(r.accion_idx)): r for r in reaccion.itertuples(index=False)}
print(f'Tabla de reacción: {len(REACCION)} celdas OK')
reaccion.pivot(index='estado_afectivo', columns='accion_idx', values='p_cierre')


# ---------- Modificadores (Tabla 5.11.b) ----------
modificadores = pd.read_csv(DATOS_DIR / 'modificadores_reaccion.csv')
IDS_ESPERADOS = ['M1', 'M2', 'M3', 'M4', 'M6', 'M7', 'M8']   # M5 se retiró (Sección 5.4.6)

assert list(modificadores['id']) == IDS_ESPERADOS, f'Modificadores: {list(modificadores["id"])}'
print(f'Modificadores: {len(modificadores)} OK')
modificadores[['id', 'condicion', 'acciones']]


DIM_ESTADO = 1 + len(TIPOS_TURISMO) + len(ESTADOS_AFECTIVOS) + 4 + len(ACTIVOS) + 1   # = 21 (incluye turno)

NOMBRES_DIM = (['segmento_intl'] + [f'tipo_{t}' for t in TIPOS_TURISMO] + [f'emo_{e}' for e in ESTADOS_AFECTIVOS]
               + ['etapa_aida_norm', 'presupuesto_norm', 'distancia_cat_norm', 'distancia_haversine_norm']
               + [f'act_{a}' for a in ACTIVOS] + ['turno_norm'])
assert len(NOMBRES_DIM) == DIM_ESTADO == 21

def vector_estado(semilla, estado_afectivo, etapa_aida, turno_siguiente=1):
    """Vector de observación s_t (Tabla 5.4) para una semilla y la situación actual del turista.
    semilla: fila de `semillas` (pd.Series). estado_afectivo y etapa_aida: valores actuales (str).
    turno_siguiente: número del turno en que se tomará la próxima acción (1..TURNO_LIMITE)."""
    v = np.zeros(DIM_ESTADO, dtype=np.float32)
    v[0] = 1.0 if semilla['segmento'] == 'internacional' else 0.0
    v[1 + TIPOS_TURISMO.index(semilla['tipo_turismo'])] = 1.0
    v[6 + ESTADOS_AFECTIVOS.index(estado_afectivo)] = 1.0
    v[11] = AIDA_NORM[etapa_aida]
    v[12] = PRESUPUESTO_NORM[semilla['presupuesto']]
    v[13] = DISTANCIA_CAT_NORM[semilla['distancia_categoria_calc']]
    v[14] = semilla['distancia_haversine_km'] / DIST_MAX_KM
    for a in semilla['activos_lista']:
        v[15 + ACTIVOS.index(a)] = 1.0
    v[20] = min(turno_siguiente, TURNO_LIMITE) / TURNO_LIMITE
    return v

# Verificación con CR-TUR-001 (Tabla 5.5): mismo vector salvo la banda de distancia recalculada y el turno
s1 = semillas.set_index('id').loc['CR-TUR-001']
v1 = vector_estado(s1, s1['estado_afectivo_inicial'], s1['etapa_aida_inicial'])
assert v1.shape == (21,) and v1.min() >= 0 and v1.max() <= 1
assert abs(v1[14] - 0.73) < 0.01, 'distancia_haversine_norm de CR-TUR-001 debería ser ~0.73'
assert abs(v1[20] - 1 / 12) < 1e-6, 'turno_norm inicial debería ser 1/12'
pd.Series(v1, index=NOMBRES_DIM).round(2).to_frame('CR-TUR-001')


DESENLACES_CIERRE = ['conversion', 'avance_parcial', 'rechazo', 'abandono']   # orden de PESOS

def activo_a_enviar(semilla):
    """Regla de la acción 4 (Tabla 5.8): primer activo disponible en el orden fijo; None si solo hay texto."""
    for a in ORDEN_ACTIVOS_ENVIO:
        if a in semilla['activos_lista']:
            return a
    return None

def aplicar_turno(estado_af, accion, semilla, etapa_aida, turno, rng):
    """Un turno del turista simulado (Sección 5.4.6). Devuelve un dict con la transición."""
    out = {'estado_siguiente': estado_af, 'delta_aida': 0, 'p_cierre': 0.0, 'cerrado': False, 'desenlace': None,
           'violacion_trust_gate': False, 'enmascarada': False, 'activo_enviado': None, 'modificadores': [],
           'probs': {}}

    # (1) M1 — oferta en trayecto inviable: desenlace forzado, detiene la evaluación
    if accion in ACCIONES_OFERTA and semilla['inviable']:
        out.update(estado_siguiente='frustrado', p_cierre=M1_P_CIERRE, violacion_trust_gate=True,
                   modificadores=['M1'], probs={'recomendacion_inviable': 1.0})
        if rng.random() < M1_P_CIERRE:
            out.update(cerrado=True, desenlace='recomendacion_inviable')
        return out

    # (2) M3 — activo no disponible: acción enmascarada, detiene la evaluación
    if accion == ACCION_ENVIAR_ACTIVO:
        out['activo_enviado'] = activo_a_enviar(semilla)
        if out['activo_enviado'] is None:
            out.update(enmascarada=True, violacion_trust_gate=True, modificadores=['M3'])
            return out

    # Celda base de la tabla de reacción (Tabla 5.11.a)
    fila = REACCION[(estado_af, accion)]
    p = float(fila.p_cierre) * ESCALA_P_CIERRE
    pesos = {d: float(getattr(fila, w)) for d, w in zip(DESENLACES_CIERRE, PESOS)}
    out.update(estado_siguiente=fila.estado_siguiente, delta_aida=int(fila.delta_aida))

    def transferir(origen, destino, puntos):
        q = min(puntos, pesos[origen])
        pesos[origen] -= q
        pesos[destino] += q

    # (3) M2 — oferta por encima del presupuesto declarado
    if accion in ACCIONES_OFERTA and NIVEL_PRESUPUESTO[PRESUPUESTO_DE_OFERTA[accion]] > NIVEL_PRESUPUESTO[semilla['presupuesto']]:
        p *= M2_FACTOR
        transferir('conversion', 'rechazo', M2_TRANSFERENCIA)
        out['violacion_trust_gate'] = True
        out['modificadores'].append('M2')
    # (4) M6 — etapa AIDA ya en acción y se ofrece paquete
    if accion in ACCIONES_OFERTA and etapa_aida == 'accion':
        p *= M6_FACTOR
        pesos['conversion'] += M6_PUNTOS_CONVERSION
        out['modificadores'].append('M6')
    # (5) M7 — cerca del límite de turnos
    if turno >= TURNO_MODIFICADOR_M7:
        p *= M7_FACTOR
        out['modificadores'].append('M7')
    # (6) M4 — ubicación enviada a desconfiado o ansioso
    if accion == ACCION_ENVIAR_ACTIVO and out['activo_enviado'] == 'ubicacion' and estado_af in ('desconfiado', 'ansioso'):
        p *= M4_FACTOR
        transferir('abandono', 'conversion', M4_TRANSFERENCIA)
        out['modificadores'].append('M4')
    # (7) tope
    p = 1.0 if accion == ACCION_CERRAR else min(p, P_CIERRE_TOPE)
    # (8) M8 — prior de la semilla (decisión D1)
    if semilla['resultado_esperado'] in pesos:
        pesos[semilla['resultado_esperado']] += PRIOR_PUNTOS
        out['modificadores'].append('M8')
    # (9) normalización y muestreo
    total = sum(pesos.values())
    probs = np.array([pesos[d] / total for d in DESENLACES_CIERRE])
    out['p_cierre'] = p
    out['probs'] = dict(zip(DESENLACES_CIERRE, probs.tolist()))
    if rng.random() < p:
        out.update(cerrado=True, desenlace=str(rng.choice(DESENLACES_CIERRE, p=probs)))
    return out

# ---- Pruebas de la dinámica contra casos conocidos ----
rng = np.random.default_rng(0)
S = semillas.set_index('id')

t = aplicar_turno('frustrado', 1, S.loc['CR-TUR-005'], 'atencion', 1, rng)          # inviable + oferta -> M1
assert t['modificadores'] == ['M1'] and t['violacion_trust_gate'] and t['estado_siguiente'] == 'frustrado'

t = aplicar_turno('entusiasmado', 4, S.loc['CR-TUR-001'], 'atencion', 1, rng)      # solo texto -> M3
assert t['enmascarada'] and t['modificadores'] == ['M3']

t = aplicar_turno('entusiasmado', 3, S.loc['CR-TUR-001'], 'atencion', 1, rng)      # presupuesto bajo, oferta alta -> M2
assert 'M2' in t['modificadores'] and abs(t['p_cierre'] - 0.45 * ESCALA_P_CIERRE * 1.5) < 1e-9

t = aplicar_turno('desconfiado', 4, S.loc['CR-TUR-004'], 'interes', 1, rng)        # ubicación a desconfiado -> M4
assert t['activo_enviado'] == 'ubicacion' and 'M4' in t['modificadores'] and abs(t['p_cierre'] - 0.15 * ESCALA_P_CIERRE * 0.7) < 1e-9

t = aplicar_turno('indeciso', 4, S.loc['CR-TUR-002'], 'interes', 11, rng)          # turno 11 -> M7
assert 'M7' in t['modificadores'] and t['estado_siguiente'] == 'entusiasmado' and t['delta_aida'] == 1

t = aplicar_turno('ansioso', 6, S.loc['CR-TUR-003'], 'interes', 1, rng)            # cerrar: siempre cierra
assert t['cerrado'] and t['p_cierre'] == 1.0

# Frecuencia empírica de desenlaces en una celda con prior: CR-TUR-001 (conversión), entusiasmado + oferta baja
res = [aplicar_turno('entusiasmado', 1, S.loc['CR-TUR-001'], 'interes', 2, rng)['desenlace'] for _ in range(20000)]
freq = pd.Series([r for r in res if r]).value_counts(normalize=True).round(3)
print('Dinámica OK. Desenlaces al cerrar (entusiasmado, oferta baja, prior=conversión):')
print(freq.to_dict(), f'| esperado: conv (75+{PRIOR_PUNTOS})/{100+PRIOR_PUNTOS}={(75+PRIOR_PUNTOS)/(100+PRIOR_PUNTOS):.2f}, '
      f'avance {20/(100+PRIOR_PUNTOS):.2f}, abandono {5/(100+PRIOR_PUNTOS):.2f}')

class BatsuEnv(gym.Env):
    """Ambiente conversacional BATSÚ (Sección 5.4). Un episodio = una conversación con un turista simulado."""
    metadata = {'render_modes': []}

    def __init__(self, semillas_df):
        super().__init__()
        self.semillas = semillas_df.set_index('id')
        self.ids = list(self.semillas.index)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(DIM_ESTADO,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACCIONES))
        self.rng = np.random.default_rng()

    # ---- ciclo de vida del episodio ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sid = options['id'] if options and 'id' in options else self.ids[self.rng.integers(len(self.ids))]
        self.semilla = self.semillas.loc[sid]
        self.id_semilla = sid
        self.estado_af = self.semilla['estado_afectivo_inicial']
        self.etapa_idx = ETAPAS_AIDA.index(self.semilla['etapa_aida_inicial'])
        self.turno = 0
        self.desenlace = None
        return self._obs(), {'id': sid}

    def _obs(self):
        return vector_estado(self.semilla, self.estado_af, ETAPAS_AIDA[self.etapa_idx], self.turno + 1)

    def mascara_acciones(self):
        """Máscara del Trust Gate (Tabla 5.12): True = acción permitida."""
        m = np.ones(len(ACCIONES), dtype=bool)
        for a in ACCIONES_OFERTA:
            if self.semilla['inviable'] or NIVEL_PRESUPUESTO[PRESUPUESTO_DE_OFERTA[a]] > NIVEL_PRESUPUESTO[self.semilla['presupuesto']]:
                m[a] = False
        if activo_a_enviar(self.semilla) is None:
            m[ACCION_ENVIAR_ACTIVO] = False
        return m

    def step(self, action):
        action = int(action)
        assert self.desenlace is None, 'El episodio ya terminó; llame a reset()'
        self.turno += 1
        estado_antes, etapa_antes = self.estado_af, self.etapa_idx
        t = aplicar_turno(self.estado_af, action, self.semilla, ETAPAS_AIDA[self.etapa_idx], self.turno, self.rng)

        # Transición afectiva y AIDA (retroceso posible; se acota a [0, 3])
        self.estado_af = t['estado_siguiente']
        self.etapa_idx = int(np.clip(self.etapa_idx + t['delta_aida'], 0, len(ETAPAS_AIDA) - 1))
        avanzo = self.etapa_idx > etapa_antes

        # Recompensa densa (Tabla 5.9); juez = 0 en esta versión (D6)
        r_shaping = R_AVANCE_AIDA if avanzo else R_ESTANCAMIENTO
        r_trust = R_VIOLACION_TRUST_GATE if t['violacion_trust_gate'] else 0.0
        r_juez = 0.0
        reward = r_shaping + r_trust + r_juez

        # Terminación (Tabla 5.11) y recompensa terminal (Tabla 5.10)
        terminated = truncated = False
        r_terminal = 0.0
        if t['cerrado']:
            self.desenlace = t['desenlace']
        elif self.turno >= TURNO_LIMITE:
            self.desenlace = 'finalizacion_limite_turnos'
        if self.desenlace is not None:
            r_terminal = R_TERMINAL[self.desenlace]
            reward += r_terminal
            terminated = self.desenlace in DESENLACES_TERMINALES
            truncated = not terminated

        info = {'id': self.id_semilla, 'turno': self.turno, 'accion': action, 'accion_nombre': ACCIONES[action],
                'estado_antes': estado_antes, 'estado_despues': self.estado_af,
                'etapa_antes': ETAPAS_AIDA[etapa_antes], 'etapa_despues': ETAPAS_AIDA[self.etapa_idx],
                'p_cierre': t['p_cierre'], 'modificadores': '|'.join(t['modificadores']),
                'activo_enviado': t['activo_enviado'], 'violacion_trust_gate': t['violacion_trust_gate'],
                'enmascarada': t['enmascarada'], 'r_shaping': r_shaping, 'r_trust': r_trust, 'r_juez': r_juez,
                'r_terminal': r_terminal, 'desenlace': self.desenlace}
        return self._obs(), float(reward), terminated, truncated, info


env = BatsuEnv(semillas)
obs, info = env.reset(seed=1, options={'id': 'CR-TUR-001'})
assert env.observation_space.contains(obs)
print('Ambiente creado ·', env.observation_space, '·', env.action_space)
print('Máscara Trust Gate para CR-TUR-001 (presupuesto bajo, solo texto):', dict(zip(ACCIONES, map(bool, env.mascara_acciones()))))


def politica_heuristica(env, epsilon=0.0, rng=None):
    """Acción esperada de la Tabla 4.2 para el estado afectivo actual; con probabilidad epsilon, acción aleatoria."""
    rng = rng or env.rng
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(len(ACCIONES)))
    e = env.estado_af
    if e == 'entusiasmado':
        return {'bajo': 1, 'medio': 2, 'alto': 3}[env.semilla['presupuesto']]   # oferta acorde al presupuesto
    if e in ('indeciso', 'desconfiado'):
        if activo_a_enviar(env.semilla) is not None:
            return ACCION_ENVIAR_ACTIVO
        return {'bajo': 1, 'medio': 2, 'alto': 3}[env.semilla['presupuesto']]   # sin activos: reducir opciones = oferta acorde

    return 0   # ansioso, frustrado: solicitar información

def politica_aleatoria(env, epsilon=None, rng=None):
    return int((rng or env.rng).integers(len(ACCIONES)))

def correr_episodio(env, politica, epsilon=0.0, seed=None, id_semilla=None):
    """Corre un episodio completo y devuelve la lista de transiciones (dicts de info + obs)."""
    obs, _ = env.reset(seed=seed, options={'id': id_semilla} if id_semilla else None)
    trans = []
    while True:
        a = politica(env, epsilon)
        obs2, r, term, trunc, info = env.step(a)
        trans.append({**info, 'reward': r, 'obs': obs, 'obs_siguiente': obs2, 'terminated': term, 'truncated': trunc})
        obs = obs2
        if term or trunc:
            return trans

def resumen(env, politica, epsilon, n=300, seed=123):
    rng = np.random.default_rng(seed)
    longitudes, retornos, desenlaces = [], [], []
    for i in range(n):
        tr = correr_episodio(env, politica, epsilon, seed=int(rng.integers(1e9)))
        longitudes.append(len(tr)); retornos.append(sum(t['reward'] for t in tr)); desenlaces.append(tr[-1]['desenlace'])
    d = pd.Series(desenlaces).value_counts(normalize=True).round(2).to_dict()
    return {'turnos_prom': round(float(np.mean(longitudes)), 1), 'retorno_prom': round(float(np.mean(retornos)), 2), 'desenlaces': d}
