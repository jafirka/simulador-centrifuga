import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math
import copy
import json
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import base64
import io




# ==========================================
# 1️⃣ TUS CLASES
# ==========================================

class Damper:
    def __init__(self, nombre, pos, kx, ky, kz, cx, cy, cz):
        self.nombre = nombre
        self.pos = np.array(pos, dtype=float)
        self.kx, self.ky, self.kz = kx, ky, kz
        self.cx, self.cy, self.cz = cx, cy, cz

    def get_matriz_T(self, cg):
        d = self.pos - np.array(cg)
        lx, ly, lz = d
        return np.array([

            [1, 0, 0,    0,   lz,  -ly], # ux: Traslación X + lz*Ry - ly*Rz
            [0, 1, 0,  -lz,    0,   lx], # uy: Traslación Y - lz*Rx + lx*Rz
            [0, 0, 1,   ly,  -lx,    0]  # uz: Traslación Z + ly*Rx - lx*Ry
        ])

    def get_matriz_K(self): return np.diag([self.kx, self.ky, self.kz])
    def get_matriz_C(self): return np.diag([self.cx, self.cy, self.cz])

class SimuladorCentrifuga:
    def __init__(self, config):
        self.pos_sensor = np.array(config["sensor"]["pos_sensor"])
        # --- Parámetros de la Placa ---
        p = config['placa']
        l_a, l_b, esp = p['lado_a'], p['lado_b'], p['espesor']
        dist_A, dist_B = p.get('dist_A', 0), p.get('dist_B', 0)
        r = p['radio_agujero']
        rho = 7850 # kg/m^3 (Acero)

	# --- Mapeo dinámico de dimensiones a ejes X, Y, Z ---
        # El espesor siempre va en el eje vertical. 
        # Los lados a y b se reparten en los ejes restantes.
        dx, dy, dz = l_a, esp, l_b
        pos_placa = [dist_A, 0, dist_B]
        self.dims = {'x': dx, 'y': dy, 'z': dz}

        # 1. Masa: Masa total - Masa del agujero
        m_total = (dx * dy * dz) * rho
        m_agujero = (math.pi * r**2 * esp) * rho
        self.m_placa = m_total - m_agujero

	# 2. Inercias respecto al CG de la placa
        # Eje Y (Eje de giro / Polar del agujero)
        Iy_b = (1/12) * m_total * (l_a**2 + l_b**2)
        Iy_a = (1/2) * m_agujero * r**2
        
        # Eje X (Diametral del agujero)
        Ix_b = (1/12) * m_total * (l_b**2 + esp**2)
        Ix_a = (1/4) * m_agujero * (r**2 + (esp**2)/3)
        
        # Eje Z (Diametral del agujero)
        Iz_b = (1/12) * m_total * (l_a**2 + esp**2)
        Iz_a = (1/4) * m_agujero * (r**2 + (esp**2)/3)

        self.I_placa = [Ix_b - Ix_a, Iy_b - Iy_a, Iz_b - Iz_a]
        
        # Posición del componente (usando dist_A y dist_B en el plano XZ)
        self.pos_placa = [p.get('dist_A', 0), 0, p.get('dist_B', 0)]


        # --- Componentes ---
        self.componentes = {
            "placa": {"m": self.m_placa, "pos": pos_placa, "I": self.I_placa},
            "cesto": config['componentes']['cesto'],
            "bancada": config['componentes']['bancada'],
            "motor": config['componentes']['motor']
        }


        # --- Excitación ---
        self.excitacion = config['excitacion']

        # --- Dampers ---
        self.dampers = []
        for d_conf in config['dampers']:
            nombre_instancia = d_conf.get('nombre', 'unnamed')
            
            # Buscamos las propiedades (kx, ky, etc.)
            # Prioridad 1: Que ya vengan en el diccionario del damper
            # Prioridad 2: Buscarlas en tipos_dampers usando el campo 'tipo'
            if 'kx' in d_conf:
                self.dampers.append(Damper(nombre_instancia, d_conf['pos'], 
                                           d_conf['kx'], d_conf['ky'], d_conf['kz'], 
                                           d_conf['cx'], d_conf['cy'], d_conf['cz']))
            else:
                tipo_nombre = d_conf['tipo']
                tipo_vals = config['tipos_dampers'][tipo_nombre]
                self.dampers.append(Damper(tipo_nombre, d_conf['pos'], **tipo_vals))

    def obtener_matriz_sensor(self, cg_global):
        r_p = self.pos_sensor - cg_global
        return np.array([
            [1, 0, 0, 0,  r_p[2], -r_p[1]],
            [0, 1, 0, -r_p[2], 0,  r_p[0]],
            [0, 0, 1,  r_p[1], -r_p[0], 0]
        ])

    def armar_matrices(self):

        
        m_total = sum(c["m"] for c in self.componentes.values())
        cg_global = sum(c["m"] * np.array(c["pos"]) for c in self.componentes.values()) / m_total

        M, I_global = np.zeros((6, 6)), np.zeros((3, 3))
        
        for nombre, c in self.componentes.items():
            m_c = c["m"]
            p_c = np.array(c["pos"])
            # Inercia local (convertir a matriz 3x3 si es lista)
            #I_local = np.diag(c["I"]) if isinstance(c["I"], list) else np.array(c["I"])

            I_local = np.array(c["I"], dtype=float)

            # Vector desde el CG global al CG del componente
            d = p_c - cg_global
        
            # Teorema de Steiner (Ejes Paralelos) en forma matricial
            # I_global = sum( I_local + m * [ (d·d)diag(1) - (d ⊗ d) ] )
            term_steiner = m_c * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

            # Verificación de simetría real antes de sumar
            matriz_c = I_local + term_steiner
            if not np.allclose(matriz_c, matriz_c.T, atol=1e-5):
                st.error(f"Asimetría detectada en componente: {nombre}")
                # Esto nos dirá si es I_local o Steiner el culpable
                st.write(f"Asimetría I_local: {np.max(np.abs(I_local - I_local.T))}")
                st.write(f"Asimetría Steiner: {np.max(np.abs(term_steiner - term_steiner.T))}")

            I_global += (I_local + term_steiner)

        M[0:3, 0:3], M[3:6, 3:6] = np.eye(3) * m_total, I_global

        K, C = np.zeros((6, 6)), np.zeros((6, 6))
        K += np.eye(6) * 1e-6
        for damper in self.dampers:
            T = damper.get_matriz_T(cg_global)
            K += T.T @ damper.get_matriz_K() @ T
            C += T.T @ damper.get_matriz_C() @ T

        # 1. Masa total
        if m_total <= 0:
            st.error("❌ Error Crítico: La masa total es cero o negativa.")

        # 2. Determinante de M (Corregido el error de sintaxis)
        det_M = np.linalg.det(M)
        if abs(det_M) < 1e-9:
            st.warning(f"⚠️ Determinante de M muy bajo ({det_M:.2e}): El sistema es físicamente imposible o singular.")

        # 3. Inercia Definida Positiva (Cholesky)
        try:
            np.linalg.cholesky(I_global) 
        except np.linalg.LinAlgError:
            st.error("🚨 ¡Inestabilidad Numérica en I_global!")
            evs = np.linalg.eigvals(I_global)
            st.write("Autovalores de I_global (deben ser todos > 0):", evs)

        # 4. Condicionamiento
        cond_M = np.linalg.cond(M)
        if cond_M > 1e12:
            st.warning(f"⚠️ Matriz de Masa mal condicionada (Cond: {cond_M:.2e}).")

        return M, K, C, cg_global



    def calcular_frecuencias_naturales(self):
        # Todo este bloque debe tener la misma sangría inicial (4 espacios)
        M, K, C, _ = self.armar_matrices()

        # Cálculo del problema de autovalores generalizado
        # K * v = λ * M * v
        evals, evecs = linalg.eigh(K, M)
        
        # Limpieza de valores por precisión numérica
        evals = np.maximum(evals, 0)
        
        # Frecuencias angulares (rad/s)
        w_n = np.sqrt(evals)
        
        # Convertir a Hz y a RPM
        f_hz = w_n / (2 * np.pi)
        f_rpm = f_hz * 60

        return f_rpm, evecs



# ==========================================
# 2️⃣ LÓGICA DE CÁLCULO
# ==========================================

def ejecutar_barrido_rpm(modelo, rpm_range, d_idx):

    M, K, C, cg_global = modelo.armar_matrices()
    T_sensor = modelo.obtener_matriz_sensor(cg_global)

    # --- Preparación damper específico ---
    damper_d = modelo.dampers[d_idx]
    T_damper = damper_d.get_matriz_T(cg_global)
    ks = [damper_d.kx, damper_d.ky, damper_d.kz]
    cs = [damper_d.cx, damper_d.cy, damper_d.cz]

    ex = modelo.excitacion
    dist = ex['distancia_eje']

    acel_cg = {"x": [], "y": [], "z": []}
    D_fuerza = {"x": [], "y": [], "z": []}
    vel_cg  = {"x": [], "y": [], "z": []}
    D_desp  = {"x": [], "y": [], "z": []}
    S_desp = {"x": [], "y": [], "z": []}
    S_vel  = {"x": [], "y": [], "z": []}
    S_acel = {"x": [], "y": [], "z": []}

    for rpm in rpm_range:
        w = rpm * 2 * np.pi / 60
        F0 = ex['m_unbalance'] * ex['e_unbalance'] * w**2
        
        # 2. Inicialización del vector de excitación F (6 DOFs: Fx, Fy, Fz, Mx, My, Mz)
        F = np.zeros(6, dtype=complex)


        # =========================================================================
        # NOTA TÉCNICA SOBRE LA EXCITACIÓN (EJE Z HORIZONTAL)
        # =========================================================================
        # Para garantizar la simetría dinámica en los apoyos, la fuerza debe 
        # aplicarse respecto al EJE DE ROTACIÓN REAL (0, 0 en el plano X-Y).
        #
        # Si el CG_global está desplazado de este eje (excentricidad lateral), 
        # la fuerza centrífuga genera momentos adicionales (Mx, My, Mz) 
        # referidos al CG que el simulador debe resolver.
        #
        # Brazos de palanca desde el CG al punto de aplicación (en el eje):
        # lx = 0 - cg_global[0] 
        # ly = 0 - cg_global[1]
        # lz = dist - cg_global[2]
        #
        # Esto corrige el "conflicto de fases" y restaura la simetría en los 
        # resultados de los dampers cuando el sistema es geométricamente espejo.
        # =========================================================================

		# --- NUEVA LÓGICA PARA EJE DE ROTACIÓN VERTICAL (Y) ---
        lx_exc = -cg_global[0]

        ly_exc = ex['distancia_eje'] - cg_global[1] # 'dist' ahora es la altura sobre el CG

        lz_exc = -cg_global[2]

        F = np.array([
          F0,                     # Fx (Centrífuga en X)
          0,                      # Fy (Nula en el eje de rotación axial)
          -1j * F0,               # Fz (Centrífuga en Z - desfase 90°)
          - (-1j * F0) * ly_exc,  # Mx = Fy*lz - Fz*ly  -> (0 - Fz*ly)
          F0 * lz_exc - (-1j * F0) * lx_exc, # My = Fz*lx - Fx*lz (Momento Torsional en el eje Y)
          F0 * ly_exc             # Mz = Fx*ly - Fy*lx  -> (Fx*ly - 0)
        ])


        # Resolver el sistema: Z * X = F
        Z = -w**2 * M + 1j*w * C + K
        X = linalg.solve(Z, F)
        # --- CG: aceleración y velocidad ---
        for i, eje in enumerate(["x", "y", "z"]):
          acel_cg[eje].append((w**2) * np.abs(X[i])/9.81)
          vel_cg[eje].append(w * np.abs(X[i]) * 1000)

        # --- Damper: desplazamiento y fuerza ---
        X_damper = T_damper @ X
        for i, eje in enumerate(["x", "y", "z"]):
          D_desp[eje].append(np.abs(X_damper[i]) * 1000)
          f_comp = (ks[i] + 1j * w * cs[i]) * X_damper[i]
          D_fuerza[eje].append(np.abs(f_comp))

        # --- Sensor: desplazamiento y fuerza ---
        U_sensor = T_sensor @ X
        for i, eje in enumerate(["x", "y", "z"]):
          # desplazamiento [mm]
          S_desp[eje].append(np.abs(U_sensor[i]) * 1000)
          # velocidad [mm/s]
          S_vel[eje].append(w * np.abs(U_sensor[i]) * 1000)
          # aceleración [g]
          S_acel[eje].append((w**2) * np.abs(U_sensor[i])/9.81)
    
    return rpm_range, D_desp, D_fuerza, acel_cg, vel_cg, S_desp, S_vel, S_acel, X_damper



def calcular_tabla_fuerzas(modelo, rpm_obj):

    M, K, C, cg_global = modelo.armar_matrices()
    m_total = sum(c["m"] for c in modelo.componentes.values())
    peso_total = m_total * 9.81
    
    n_d = len(modelo.dampers)
    if n_d == 0: return pd.DataFrame()

    # --- 1. REPARTO ESTÁTICO (Consistente con Vertical = Y) ---
    A = np.zeros((3, n_d))
    b = np.array([peso_total, 0, 0])
    for i, d in enumerate(modelo.dampers):
        rx = d.pos[0] - cg_global[0]
        rz = d.pos[2] - cg_global[2]
        A[0, i] = 1        # Suma de fuerzas en Y
        A[1, i] = rz       # Momento en X (Brazo Z)
        A[2, i] = -rx      # Momento en Z (Brazo X)

    reacciones_estaticas = np.linalg.pinv(A) @ b

    # --- 2. CÁLCULO DINÁMICO LLAMANDO AL BARRIDO ---
    # Llamamos al barrido para cada damper para obtener sus fuerzas
    resumen = []
    
    for i, d in enumerate(modelo.dampers):
        # Ejecutamos el barrido solo para la RPM objetivo y para este damper específico
        # Pasamos [rpm_obj] como lista para que el bucle for del barrido funcione
        _, D_desp, D_fuerza, *_ = ejecutar_barrido_rpm(modelo, [rpm_obj], d_idx=i)
        
        # Como solo enviamos una RPM, los resultados están en el índice [0] de las listas
        f_din_x = D_fuerza["x"][0]
        f_din_y = D_fuerza["y"][0]
        f_din_z = D_fuerza["z"][0]
        
        f_est_y = reacciones_estaticas[i]
        
        resumen.append({
            "Damper": d.nombre,
            "Carga Estática [N]": round(f_est_y, 1),
            "Dinámica X [N]": round(f_din_x, 1),
            "Dinámica Y [N]": round(f_din_y, 1),
            "Dinámica Z [N]": round(f_din_z, 1),
            "Carga TOTAL MÁX [N]": round(f_est_y + f_din_y, 1),
            "Margen Estabilidad [N]": round(f_est_y - f_din_y, 1)
        })

    return pd.DataFrame(resumen)



def graficar_fuerza_tiempo(modelo, rpm, d_idx):
    res = ejecutar_barrido_rpm(modelo, [rpm], d_idx)
    
    # IMPORTANTE: Si el barrido devuelve una lista de vectores complejos, 
    # tomamos el primero. Si devuelve solo uno, lo usamos directo.
    X_data = res[-1]
    X_target = X_data[0] if isinstance(X_data, list) else X_data

    w = rpm * 2 * np.pi / 60
    t = np.linspace(0, 2 * (2 * np.pi / w), 500)
    
    d = modelo.dampers[d_idx]
    ks = [d.kx, d.ky, d.kz]
    cs = [d.cx, d.cy, d.cz]
    f_ejes = {"x": [], "y": [], "z": []}

    for ti in t:
        fasor = np.exp(1j * w * ti)
        for i, eje in enumerate(["x", "y", "z"]):
            # La clave es que X_target[i] sea el complejo A + Bi
            term_dinamico = (ks[i] + 1j * w * cs[i]) * X_target[i] * fasor
            f_ejes[eje].append(term_dinamico.real)

    # 4. Crear la figura de Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, f_ejes["x"], label="Fuerza X (Lateral)", color="tab:blue", alpha=0.7)
    ax.plot(t, f_ejes["y"], label="Fuerza Y (Vertical)", color="tab:orange", linewidth=2.5)
    ax.plot(t, f_ejes["z"], label="Fuerza Z (Axial)", color="tab:green", alpha=0.7)
    
    ax.set_title(f"Análisis Temporal: Damper {d.nombre} a {rpm} RPM")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Fuerza Dinámica [N]")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig



def dibujar_modelo_2d(modelo, titulo="Disposición de Planta (Plano XZ)"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Obtener datos del modelo
    p = modelo.dims
    pos_p = modelo.pos_placa
    # Calculamos el CG global para graficarlo
    _, _, _, cg_global = modelo.armar_matrices()

    # 2. Dibujar la Placa de Inercia (Rectángulo)
    rect_placa = plt.Rectangle(
        (pos_p[0] - p['x']/2, pos_p[2] - p['z']/2), 
        p['x'], p['z'], 
        linewidth=1.5, edgecolor='#333333', facecolor='lightgray', alpha=0.4, label='Placa Inercia'
    )
    ax.add_patch(rect_placa)

    # 3. Dibujar el Cesto (Referencia circular)
    radio_cesto = (modelo.excitacion.get('e_unbalance', 0.625)) # Radio basado en config
    cesto_circ = plt.Circle((0, 0), radio_cesto, color='blue', fill=False, ls='--', alpha=0.5, label='Rotor/Cesto')
    ax.add_patch(cesto_circ)

    # 4. Dibujar Dampers (Puntos de apoyo)
    for i, d in enumerate(modelo.dampers):
        ax.scatter(d.pos[0], d.pos[2], marker='s', s=80, color='black', alpha=0.7, 
                   label="Dampers (Apoyos)" if i==0 else "")
        ax.text(d.pos[0], d.pos[2] + 0.08, d.nombre, fontsize=7, ha='center')

    # 5. DIBUJAR ÚNICAMENTE EL CG GLOBAL
    # Usamos una estrella dorada o roja para que sea el foco de atención
    ax.scatter(cg_global[0], cg_global[2], marker='*', s=250, color='red', 
               edgecolor='black', zorder=10, label='CG GLOBAL SISTEMA')
    
    # Etiqueta con coordenadas del CG
    ax.text(cg_global[0] + 0.1, cg_global[2] + 0.1, 
            f"CG ({cg_global[0]:.2f}, {cg_global[2]:.2f})", 
            color='red', fontweight='bold', fontsize=9)

    # 6. Punto de Giro (Origen 0,0)
    ax.scatter(0, 0, marker='+', color='blue', s=100, label='Eje de Rotación')

    # Configuración final
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel("Eje X [m]")
    ax.set_ylabel("Eje Z [m]")
    ax.set_title(titulo, pad=20)
    ax.legend(loc='upper right', fontsize='small', frameon=True)
    
    return fig


# ==========================================
# 3️⃣ ENTORNO VISUAL (INTERFAZ)
# ==========================================

def inicializar_estado_del_simulador():
    # --- INICIALIZADOR DE DATOS (Fuente de Verdad Única) ---
    if 'componentes_data' not in st.session_state:
        st.session_state.componentes_data = {
            "bancada": {"m": 1000.0, "pos": [0.0, 0.0, 0.0], "I": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]]},
            "motor": {"m": 940.0, "pos": [1.6, 0.0, 1.1], "I": [[178.0, 0, 0], [0, 392.0, 0], [0, 0, 312.0]]},
            "cesto": {"m": 1000.0, "pos": [0.0, 0.0, 0.0], "I": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]]}
        }
    if 'placa_data' not in st.session_state:
            st.session_state.placa_data = {
            "lado_a": 2.4, "lado_b": 2.4, "espesor": 0.1, 
            "radio_agujero": 0.5, "dist_A": 0.0, "dist_B": 0.0
        }
    if 'configuracion_sistema' not in st.session_state:
        st.session_state.configuracion_sistema = {
            "distancia_eje": 0.0,
            "sensor_pos": [-0.4, 0.2, 0.0],
            "diametro_cesto": 1250  # Valor por defecto (mm)
        }

    if 'dampers_prop_data' not in st.session_state:
        st.session_state.dampers_prop_data = [
            {"Tipo": "ZPVL_XXX", "kx": 1.5e6, "ky": 1.5e6, "kz": 1.5e6, "cx": 5.5e4, "cy": 5.5e4, "cz": 5.5e4},
            {"Tipo": "ZPVL_YYY", "kx": 1.5e6,  "ky": 1.5e6,  "kz": 1.5e6, "cx": 5.5e4, "cy": 5.5e4, "cz": 5.5e4}
        ]

    if 'dampers_pos_data' not in st.session_state:
        st.session_state.dampers_pos_data = [
            {"Nombre": "D1 (Frontal)", "X": -1.4, "Y": -0.4, "Z": 1.4, "Tipo": "ZPVL_XXX"},
            {"Nombre": "D2 (Frontal)", "X": 1.4, "Y":  -0.4, "Z": 1.4, "Tipo": "ZPVL_XXX"},
            {"Nombre": "D3 (Posterior)", "X": -1.4, "Y": -0.4, "Z": -1.4, "Tipo": "ZPVL_YYY"},
            {"Nombre": "D4 (Posterior)", "X": 1.4, "Y":  -0.4, "Z": -1.4, "Tipo": "ZPVL_YYY"},
        ]

# --- 1. INTERFAZ DE STREAMLIT ---
st.set_page_config(layout="wide")

# Llamamos a nuestra nueva función
inicializar_estado_del_simulador()

st.title("Simulador Interactivo de Centrífuga 300F - Departamento de Ingenieria de Riera Nadeu")
st.markdown("Modifica los valores en la barra lateral para ver el impacto en las vibraciones.")

# --- BARRA LATERAL PARA MODIFICAR VALORES ---
st.sidebar.header("Parámetros de cálculos")


# Ejemplo de cómo modificar la masa de desbalanceo y RPM
m_unbalance = st.sidebar.slider("Masa de Desbalanceo (kg)", 0.1, 8.0, 1.6)
rpm_obj = st.sidebar.number_input("RPM nominales", value=1100)

# --- SECCIÓN: PESTAÑAS ---
st.header("🧱 Configuración del Sistema")

# Contenedor para los datos procesados en los tabs
comp_editados = {} 
tab_config, tab_comp, tab_dampers, = st.tabs([ "⚙️ Configuración del Sistema", "📦 Componentes Masas/Inercias", "🛡️ Configuración de Dampers"])

# 1️⃣ CONFIGURACION DE SISTEMA
with tab_config:
    st.subheader("Configuración de Ejes y Convención")
    # 1. Leemos del "log" (session_state) para establecer el valor inicial
    distancia_eje = st.number_input(
        "Coordenada horizontal de la masa de desbalanceo (m)", 
        value=float(st.session_state.configuracion_sistema.get("distancia_eje", 0.8)),
        step=0.01,
        format="%.2f"
    )

    # --- DENTRO DE tab_config ---
    opciones_diametro = [800, 1000, 1250, 1400, 1600, 1800, 2000]

    # Simplificado: Calculamos el índice directamente en una línea
    # Al no tener 'key', el selectbox obedecerá siempre al 'index' que viene del JSON
    diametro_sel = st.selectbox(
        "Tamaño de cesto (Diámetro en mm):", 
        opciones_diametro, 
        index=opciones_diametro.index(st.session_state.configuracion_sistema.get("diametro_cesto", 1250))
    )

    # 3. Calculamos la excentricidad (Radio en metros)
    e_unbalance = (diametro_sel / 1000) / 2

       
    # --- NUEVA SECCIÓN: POSICIÓN DEL SENSOR ---
    st.text("Posición del Sensor de velocidad/aceleracion(m)")
    col_s1, col_s2, col_s3 = st.columns(3)

    sensor_actual = st.session_state.configuracion_sistema.get("sensor_pos", [0.0, 0.0, 0.0])
    with col_s1:
        sensor_x = st.number_input("X", value=float(sensor_actual[0]), step=0.1)
    with col_s2:
        sensor_y = st.number_input("Y", value=float(sensor_actual[1]), step=0.1)
    with col_s3:
        sensor_z = st.number_input("Z", value=float(sensor_actual[2]), step=0.1)

    st.divider()

# Actualizamos los valores de sistema en el session_state con lo que hay actualmente en los widgets
st.session_state.configuracion_sistema["distancia_eje"] = distancia_eje
st.session_state.configuracion_sistema["sensor_pos"] = [sensor_x, sensor_y, sensor_z] 
st.session_state.configuracion_sistema["diametro_cesto"] = diametro_sel

# 1️⃣ GESTIÓN DE COMPONENTES (Inercia 3x3 con Persistencia)
with tab_comp:
    subtabs = st.tabs(["Bancada", "Accionamiento", "Cesto", "Placa inercia"])
    
    # Mapeo de nombres para session_state
    nombres_llaves = ["bancada", "cesto", "Accionamiento"]

    for i, nombre in enumerate(nombres_llaves):
        with subtabs[i]:
            # ✅ NOTA ACLARATORIA: Solo para la primera subpestaña (Bancada)
            if i == 0:
                st.info("💡 **Nota:** La bancada debe inlcuir la masa y la inercia de la caja de rodamientos y elementos auxilaires, EXCLUYENDO la placa de inercia")
            # 1. LEER DEL LOG (Fuente de Verdad)
            datos_memoria = st.session_state.componentes_data[nombre]
            pos_actual = datos_memoria.get("pos", [0.0, 0.0, 0.0])
            
            c_m, c_p = st.columns([1, 2])
            with c_m:
                m_val = st.number_input(f"Masa {nombre} (kg)", value=float(datos_memoria.get("m", 0.0)))

            with c_p:
                cx, cy, cz = st.columns(3)
                px = cx.number_input(f"X {nombre} [m]", value=float(pos_actual[0]), format="%.3f")
                py = cy.number_input(f"Y {nombre} [m]", value=float(pos_actual[1]), format="%.3f")
                pz = cz.number_input(f"Z {nombre} [m]", value=float(pos_actual[2]), format="%.3f")
           
            st.write(f"**Matriz de Inercia (3x3) [kg·m²]**")

            # El data_editor es excelente para matrices
            df_iner_3x3 = st.data_editor(
                np.array(datos_memoria["I"]),
                key=f"editor_matriz_{nombre}", # El editor sí necesita key para ser interactivo
                use_container_width=True
            )
            
            # 2. ACTUALIZAR LOG (Sincronización inmediata)
            st.session_state.componentes_data[nombre] = {
                "m": m_val, 
                "pos": [px, py, pz], 
                "I": df_iner_3x3.tolist() if isinstance(df_iner_3x3, np.ndarray) else df_iner_3x3
            }

# 2. ✅ Datos de la Placa de Inercia
with subtabs[3]:
    st.write("### Parámetros Geométricos de la Placa")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # LEER DEL LOG: Eliminamos las 'key' de los inputs para que no bloqueen la carga del JSON
        lado_a = st.number_input("Lado A [m]", 
                                value=float(st.session_state.placa_data.get("lado_a", 2.4)), 
                                step=0.1, format="%.2f")
        
        lado_b = st.number_input("Lado B [m]", 
                                value=float(st.session_state.placa_data.get("lado_b", 2.4)), 
                                step=0.1, format="%.2f")
    
    with col_g2:
        espesor = st.number_input("Espesor [m]", 
                                    value=float(st.session_state.placa_data.get("espesor", 0.1)), 
                                    step=0.01, format="%.3f")
        radio_agujero = st.number_input("Radio Agujero [m]", 
                                        value=float(st.session_state.placa_data.get("radio_agujero", 0.5)), 
                                        step=0.05, format="%.2f")

    st.write("### Posición del Centro de la Placa")
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        dist_A = st.number_input(f"Desfase en x (dist_A)", 
                                 value=float(st.session_state.placa_data.get("dist_A", 0.0)), 
                                 step=0.1)
    with col_p2:
        dist_B = st.number_input(f"Desfase en z (dist_B)", 
                                 value=float(st.session_state.placa_data.get("dist_B", 0.0)), 
                                 step=0.1)

    # ✅ ACTUALIZACIÓN DEL LOG (Sincronización inmediata)
    # Al estar fuera de los 'with col', se ejecuta siempre que cambie cualquier valor
    st.session_state.placa_data.update({
        "lado_a": lado_a,
        "lado_b": lado_b,
        "espesor": espesor,
        "radio_agujero": radio_agujero,
        "dist_A": dist_A,
        "dist_B": dist_B
    })



# 2️⃣ GESTIÓN DE DAMPERS
with tab_dampers:
    st.write("### 1. Definición de Propiedades por Tipo")
    
    # Editor de Propiedades: Sincronización directa con el Log
    df_prop_editada = st.data_editor(
        st.session_state.dampers_prop_data,
        key="editor_tipos_nombres",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic", # Permitimos que el usuario defina nuevos tipos
        column_config={
            "Tipo": st.column_config.TextColumn("Nombre del Tipo/Modelo", required=True),
            "kx": st.column_config.NumberColumn("Kx [N/m]", format="%.1e"),
            "ky": st.column_config.NumberColumn("Ky [N/m]", format="%.1e"),
            "kz": st.column_config.NumberColumn("Kz [N/m]", format="%.1e"),
        }
    )
    # SOLO actualizamos el session_state si el widget ha cambiado realmente
    # Esto evita que el widget "pise" al JSON recién cargado
    if st.session_state.get("editor_tipos_nombres"):
        st.session_state.dampers_prop_data = df_prop_editada

    # Extraemos la lista de tipos para el desplegable de la siguiente tabla
    # Usamos list(set(...)) para evitar duplicados si el usuario se equivoca
    df_prop = pd.DataFrame(df_prop_editada)
    lista_tipos_disponibles = df_prop["Tipo"].dropna().unique().tolist()

    st.write("### 2. Ubicación de los Dampers")
    
    # Editor de Posiciones
    res_pos_editor = st.data_editor(
        st.session_state.dampers_pos_data,
        num_rows="dynamic", 
        key="pos_dampers_editor_v2", 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nombre": st.column_config.TextColumn("Identificador (ej. D1)", required=True),
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo de Damper", 
                options=lista_tipos_disponibles,
                required=True
            ),
            "X": st.column_config.NumberColumn("X [m]", format="%.3f"),
            "Y": st.column_config.NumberColumn("Y [m]", format="%.3f"),
            "Z": st.column_config.NumberColumn("Z [m]", format="%.3f"),
        }
    )
    # Sincronizamos con el Log Maestro
    st.session_state.dampers_pos_data = res_pos_editor

    # ✅ PROCESAMIENTO FINAL (Para el motor de cálculo)
    # Creamos 'dampers_finales' uniendo ambas tablas
    dampers_finales = []
    df_pos = pd.DataFrame(res_pos_editor)
    
    if not df_pos.empty and not df_prop.empty:
        df_prop_indexed = df_prop.set_index("Tipo")
        
        for _, row in df_pos.iterrows():
            tipo_sel = row.get("Tipo")
            # Seguridad: Solo procesamos si el tipo existe en la tabla de propiedades
            if tipo_sel and tipo_sel in df_prop_indexed.index:
                p = df_prop_indexed.loc[tipo_sel]
                dampers_finales.append({
                    "nombre": row.get("Nombre", "Sin nombre"),
                    "pos": [row.get("X", 0.0), row.get("Y", 0.0), row.get("Z", 0.0)],
                    "tipo": tipo_sel,
                    "kx": p["kx"], "ky": p["ky"], "kz": p["kz"],
                    "cx": p.get("cx", 0.0), "cy": p.get("cy", 0.0), "cz": p.get("cz", 0.0)
                })



# 3️⃣ ENSAMBLAJE FINAL (Cálculo Base)
# Usamos las llaves del session_state para garantizar que, 
# aunque el usuario no abra una pestaña, el simulador use el último dato guardado.

config_base = {
    "excitacion": {
        "distancia_eje": st.session_state.configuracion_sistema["distancia_eje"], 
        "m_unbalance": m_unbalance, # Viene del slider de la sidebar
        "e_unbalance": e_unbalance # Valor constante de diseño
    },
    "placa": st.session_state.placa_data,
    "componentes": st.session_state.componentes_data,
    "dampers": dampers_finales, # Lista ya procesada en la pestaña anterior
    "sensor": {
        "pos_sensor": st.session_state.configuracion_sistema["sensor_pos"]
    },
    "tipos_dampers": pd.DataFrame(st.session_state.dampers_prop_data).set_index("Tipo").to_dict('index')
}


# --- SELECTOR DE DAMPER ---
# Accedemos directamente al diccionario de configuración
lista_dampers_config = config_base["dampers"] 
# Creamos las opciones para el selectbox usando el diccionario
opciones = [f"{i}: {d['tipo']} en {d['pos']}" for i, d in enumerate(lista_dampers_config)]
seleccion = st.sidebar.selectbox("Selección de damper para diagnóstico:", opciones)
# Extraemos el índice
d_idx = int(seleccion.split(":")[0])

# --- 2. INTERFAZ PARA LA PROPUESTA (Sliders) ---
st.sidebar.header("Variaciones de la Propuesta")
esp_prop = st.sidebar.slider("Espesor Propuesta [mm]", 40.0, 140.0, 100.0) / 1000
pos_x_motor_prop = st.sidebar.slider("Posición X Motor Propuesta [m]", 1.2, 1.8, 1.6)

# --- 3. CREAR CONFIGURACIÓN DINÁMICA ---
config_prop = copy.deepcopy(config_base)
config_prop["placa"]["espesor"] = esp_prop
config_prop["componentes"]["motor"]["pos"][0] = pos_x_motor_prop

# --- 4. EJECUTAR AMBAS SIMULACIONES ---
modelo_base = SimuladorCentrifuga(config_base)
modelo_prop = SimuladorCentrifuga(config_prop)

f_res_rpm, modos = modelo_base.calcular_frecuencias_naturales()
f_res_rpm_prop, modos_prop = modelo_prop.calcular_frecuencias_naturales()

# RPM de operación
rpm_range = np.linspace(10, rpm_obj*1.2, 1000)
idx_op = np.argmin(np.abs(rpm_range - rpm_obj))
rpm_range, D_desp, D_fuerza, acel_cg, vel_cg, S_desp, S_vel, S_acel, X_damper = ejecutar_barrido_rpm(modelo_base, rpm_range, d_idx)
rpm_range, desp_prop, fuerza_prop, acel_prop, vel_prop, S_desp_prop, S_vel_prop, S_acel_prop, X_damper_prop = ejecutar_barrido_rpm(modelo_prop, rpm_range, d_idx)




# ==========================================
# 📄 INTRODUCCIÓN Y MEMORIA DE CÁLCULO
# ==========================================
st.markdown(f"""
### 📋 Resultados
---
""")

st.markdown("""
    <style>
    @media print {
        /* Ocultar barra lateral y botones al imprimir */
        [data-testid="stSidebar"], .stButton, header, footer {
            display: none !important;
        }
        /* Ajustar el ancho del contenedor principal */
        .main .block-container {
            max-width: 100%;
            padding: 1rem;
        }
        /* Forzar que los gráficos no se corten entre páginas */
        .stPlotlyChart, .css-12w0qpk {
            page-break-inside: avoid;
        }
    }
    </style>
    """, unsafe_allow_html=True)



st.header("🗺️ Mapa de Disposición Física")

col_vista_base, col_vista_prop = st.columns(2)

with col_vista_base:
    st.subheader("Configuración Base")
    fig_base = dibujar_modelo_2d(modelo_base, "Planta: Configuración Base")
    st.pyplot(fig_base)
    
    # Info de CG
    _, _, _, cg_b = modelo_base.armar_matrices()
    st.info(f"**CG Base:** X:{cg_b[0]:.2f}, Y:{cg_b[1]:.2f}, Z:{cg_b[2]:.2f}")

with col_vista_prop:
    st.subheader("Configuración Propuesta")
    fig_prop = dibujar_modelo_2d(modelo_prop, "Planta: Variación Propuesta")
    st.pyplot(fig_prop)
    
    # Info de CG
    _, _, _, cg_p = modelo_prop.armar_matrices()
    st.success(f"**CG Propuesta:** X:{cg_p[0]:.2f}, Y:{cg_p[1]:.2f}, Z:{cg_p[2]:.2f}")

st.divider()
st.subheader("⏱️ Respuesta Temporal de Fuerzas")
st.info(f"Mostrando el comportamiento oscilatorio para el Damper seleccionado a {rpm_obj} RPM.")

# 1. Creamos las filas de columnas (2 columnas por fila)
fila1 = st.columns(2)
fila2 = st.columns(2)

# 2. Las unimos en una lista plana para iterar fácilmente
columnas = fila1 + fila2 

# 3. Iteramos sobre los 4 dampers (D1 a D4)
for i, col in enumerate(columnas):
    with col:
        # Obtenemos el nombre del damper para el título
        nombre_d = modelo_base.dampers[i].nombre
        
        # Generamos la figura
        # Nota: He quitado fig.update_layout porque eso es para Plotly. 
        # Si usas Matplotlib, el tamaño se controla en plt.subplots(figsize=...)
        fig = graficar_fuerza_tiempo(modelo_base, rpm_obj, i)
        
        # Mostramos en Streamlit
        st.pyplot(fig)




st.subheader("📋 Resumen de Cargas por Apoyo")
df_cargas = calcular_tabla_fuerzas(modelo_base, rpm_obj)

if not df_cargas.empty:
    st.dataframe(
        df_cargas,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Carga Vert. Máx (Est+Z) [N]": st.column_config.NumberColumn(
                "Carga Vert. Máx [N]",
                help="Suma de la carga estática y la amplitud de la fuerza dinámica vertical (Z).",
                format="%.1f"
            )
        }
    )

st.markdown("""
### 💡 Guía de Interpretación de Cargas
* **Carga Estática:** Es el peso de la máquina (Bancada + Cesto) distribuido en cada apoyo según la posición del Centro de Gravedad (CG).
* **Dinámica (X, Y, Z):** Es la amplitud de la fuerza vibratoria generada por el desbalanceo a las RPM nominales.
* **Carga TOTAL MÁX:** Es la carga máxima que el damper debe soportar estructuralmente ($F_{est} + F_{din, Vertical}$). Útil para verificar la capacidad del catálogo del fabricante.
* **Margen de Estabilidad:** Es la fuerza neta mínima durante la oscilación ($F_{est} - F_{din, Vertical}$). 
    * **Si es > 0:** El apoyo siempre está en compresión (Seguro).
    * **Si es < 0:** El apoyo intenta levantarse de la base (Vuelo), lo que genera impactos, ruido y desgaste prematuro.
""")

st.markdown("""
### 💡 Gráficos
---
""")

# --- DEFINICIÓN DE EJES PARA GRÁFICOS (Pegar antes de los bucles for) ---
eje_axial = "z"
eje_vert_fisico = "y"  # P.ej: si eje es 'x', este es 'y'
eje_horiz_fisico = "x" # P.ej: si eje es 'x', este es 'z'

# Creamos la lista para iterar en los gráficos
orden_grafico = [eje_vert_fisico, eje_horiz_fisico, eje_axial]

# Diccionario de etiquetas para las leyendas
ejes_lbl = {
    eje_vert_fisico: f"({eje_vert_fisico.upper()})",
    eje_horiz_fisico: f"({eje_horiz_fisico.upper()})",
    eje_axial: f"({eje_axial.upper()})"
}

# Diccionario de colores
colores = {
    eje_vert_fisico: "tab:orange", 
    eje_horiz_fisico: "tab:blue", 
    eje_axial: "tab:green"
}


# ==========================
# 📊 GRÁFICO 1: Aceleración en el SENSOR (CORREGIDO)
# ==========================
st.subheader("Análisis de Aceleración en el Sensor")
fig, ax = plt.subplots(figsize=(10, 4))

# Usamos el orden dinámico y las etiquetas mapeadas
for eje in orden_grafico:
    ax.plot(rpm_range, S_acel[eje], color=colores[eje], label=ejes_lbl[eje])

ax.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax.set_xlabel("RPM")
ax.set_ylabel("Aceleración [g]")
ax.grid(True, alpha=0.1)
ax.legend()
plt.rcParams.update({'font.size': 10}) 
fig.tight_layout()
st.pyplot(fig, clear_figure=True)


# ==========================
# 📊 GRÁFICO 2: Velocidad en el SENSOR (REVISADO)
# ==========================
st.subheader("Respuesta en Frecuencia: Velocidad en Sensor")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for eje in orden_grafico:
    ax2.plot(rpm_range, S_vel[eje], color=colores[eje], label=ejes_lbl[eje])

ax2.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')

# Marcar resonancias teóricas
for i, f in enumerate(f_res_rpm):
    if f < rpm_range[-1]: 
        ax2.axvline(f, color='red', linestyle='--', alpha=0.3, 
                    label='Resonancia' if i == 0 else "") # Etiqueta solo una vez

ax2.set_xlabel('Velocidad de Rotación [RPM]')
ax2.set_ylabel('Velocidad [mm/s]')
ax2.grid(True, alpha=0.1)
ax2.legend()
st.pyplot(fig2)

# Inserta esto antes de una sección nueva que quieras que empiece en hoja limpia
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.subheader(f"Desplazamiento Amplitud en Damper {lista_dampers_config[d_idx]['tipo']}")


# ==========================
# 📊 GRÁFICO 3: Desplazamiento Damper
# ==========================
fig3, ax3 = plt.subplots(figsize=(10, 5))
for eje in orden_grafico:
    ax3.plot(rpm_range, D_desp[eje], color=colores[eje], label=f'{ejes_lbl[eje]}')

ax3.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax3.set_xlabel('Velocidad de Rotación [RPM]')
ax3.set_ylabel('Desplazamiento [mm]')
ax3.grid(True, alpha=0.1)
ax3.legend()
st.pyplot(fig3)

# ==========================
# 📊 GRÁFICO 4: Fuerzas Dinámicas
# ==========================
st.subheader(f"Fuerzas Dinámicas en Damper {lista_dampers_config[d_idx]['tipo']}")
fig4, ax4 = plt.subplots(figsize=(10, 5))
# Usamos el orden lógico: Radial Vertical, Radial Horizontal y Axial
for eje in orden_grafico:
    ax4.plot(rpm_range, D_fuerza[eje], color=colores[eje], label=ejes_lbl[eje])
ax4.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
# --- CORRECCIÓN DE LA ANOTACIÓN ---
# Usamos el eje vertical físico (donde realmente hay carga dinámica)
eje_v = eje_vert_fisico 
f_max_op = D_fuerza[eje_v][idx_op]
ax4.annotate(
    f'{f_max_op:.0f} N ({eje_v.upper()}) a {rpm_obj} RPM',
    xy=(rpm_range[idx_op], f_max_op),
    # Ajustamos xytext para que no se solape con la línea de la curva
    xytext=(rpm_range[idx_op] * 0.6, f_max_op * 1.15), 
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
    fontsize=10,
    fontweight='bold'
)
ax4.set_xlabel('Velocidad de Rotación [RPM]')
ax4.set_ylabel('Fuerza Transmitida [N]')
ax4.grid(True, alpha=0.1)
# Colocamos la leyenda fuera del gráfico si hay muchas líneas
ax4.legend(loc='upper right')
st.pyplot(fig4)




# Inserta esto antes de una sección nueva que quieras que empiece en hoja limpia
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.title("Simulador de variaciones de parametros")
st.subheader("Comparativa de velocidad en el Sensor")


# ==========================
# 📊 GRÁFICO 5: Comparativa Fuerzas Dinámicas
# ==========================
fig5, ax5 = plt.subplots(figsize=(10, 4))
# --- CASO BASE (Líneas punteadas o grises) ---
ax5.plot(rpm_range, S_vel["x"], color="gray", linestyle="--", alpha=0.5, label="Base X")
ax5.plot(rpm_range, S_vel["y"], color="silver", linestyle="--", alpha=0.5, label="Base Y")
ax5.plot(rpm_range, S_vel["z"], color="black", linestyle="--", alpha=0.5, label="Base Z")
# --- PROPUESTA (Colores vivos) ---
ax5.plot(rpm_range, S_vel_prop["x"], color="tab:blue", label="Propuesta X")
ax5.plot(rpm_range, S_vel_prop["y"], color="tab:orange", label="Propuesta Y")
ax5.plot(rpm_range, S_vel_prop["z"], color="tab:green", label="Propuesta Z")
ax5.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax5.set_xlabel("RPM")
ax5.set_ylabel("velocidad [mm/s]")
ax5.legend()
ax5.grid(True, alpha=0.1)
st.pyplot(fig5)

# ==========================
# 📊 GRÁFICO 6: Comparativa Fuerzas Dinámicas
# ==========================

st.subheader(f"Comparativa de fuerza vertical en el Damper {lista_dampers_config[d_idx]['tipo']}")

fig6, ax6 = plt.subplots(figsize=(10, 4))
# --- CASO BASE (Líneas punteadas o grises) ---
ax6.plot(rpm_range, D_fuerza[y], color="gray", linestyle="--", alpha=0.5, label="Base X")
# --- PROPUESTA (Colores vivos) ---
ax6.plot(rpm_range, fuerza_prop[1], color="tab:blue", label="Propuesta X")
Fy_orig_1100 = D_fuerza[1][idx_op]
Fy_prop_1100 = fuerza_prop[1][idx_op]
# --- Anotaciones ---
plt.annotate(
    f'{Fz_prop_1100:.0f} N',
    xy=(rpm_obj, Fz_prop_1100),
    xytext=(rpm_obj+80, Fz_prop_1100*1.25),
    arrowprops=dict(arrowstyle='->', color='blue'),
    color='blue'
)
plt.annotate(
    f'{Fz_orig_1100:.0f} N',
    xy=(rpm_obj, Fz_orig_1100),
    xytext=(rpm_obj+80, Fz_orig_1100*0.85),
    arrowprops=dict(arrowstyle='->', color='gray'),
    color='gray'
)
ax6.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax6.set_xlabel("RPM")
ax6.set_ylabel("Fuerza [N]")
ax6.legend()
ax6.grid(True, alpha=0.1)
st.pyplot(fig6)





# ==========================================
# 📈 ANÁLISIS DE RESONANCIA Y CONCLUSIONES
# ==========================================
st.markdown(f"""
### 📋 Informe de Análisis Dinámico
Este reporte simula el comportamiento vibratorio de una centrífuga industrial bajo condiciones de desbalanceo.
A continuación se detallan los parámetros de entrada utilizados para este análisis:

* **Masa de Desbalanceo:** {m_unbalance:.2f} kg
* **RPM de Operación:** {rpm_obj} RPM
---
""")


st.divider()
# Inserta esto antes de una sección nueva que quieras que empiece en hoja limpia
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.header("Análisis de Seguridad y Vibraciones")

# 1. Identificación de la Frecuencia Crítica (Resonancia)
# Buscamos el pico máximo en el barrido de RPM
idx_res_base = np.argmax(S_vel[eje_v])
rpm_res_base = rpm_range[idx_res_base]

col_concl1, col_concl2 = st.columns(2)

with col_concl1:
    st.write("### 🚨 Puntos Críticos (Resonancia)")
    # Mostramos la primera frecuencia natural (Modo 1)
    st.write(f"**Caso Base (Modo 1):** {f_res_rpm[0]:.0f} RPM")
    st.write(f"**Caso Base (Modo 2):** {f_res_rpm[1]:.0f} RPM")
    st.write(f"**Caso Base (Modo 3):** {f_res_rpm[2]:.0f} RPM")
    st.write(f"**Caso Base (Modo 4):** {f_res_rpm[3]:.0f} RPM")
    st.write(f"**Caso Base (Modo 5):** {f_res_rpm[4]:.0f} RPM")
    st.write(f"**Caso Base (Modo 6):** {f_res_rpm[5]:.0f} RPM")

    
    dist_min_base = abs(f_res_rpm[5] - rpm_obj)
    if dist_min_base < 150:
        # Identificamos cuál falló para dar un mensaje preciso
        st.error(f"⚠️ PELIGRO: Resonancia crítica detectada. "
                 f"Margen insuficiente (< 150 RPM) respecto a {rpm_obj} RPM.")
    else:
        st.success(f"✅ SEGURO: Todos los modos de ambos modelos mantienen un margen "
                   f"> 150 RPM respecto a la operación.")
        
    st.caption(f"Margen actual: Base {dist_min_base:.0f} RPM")

with col_concl2:
    st.write("### 📊 Cumplimiento de Norma (ISO 10816)")
    
    # Extraemos el pico máximo considerando los tres ejes para ser conservadores
    v_max_base = max(max(S_vel["x"]), max(S_vel["y"]), max(S_vel["z"]))
        
    st.write(f"**Velocidad Máx. detectada:** {v_max_base:.2f} mm/s")
    
    # Opcional: Clasificación rápida
    if v_max_base > 12.0:
        st.warning("Zona C: Vibración insatisfactoria para operación continua.")
    elif v_max_base > 8.0:
        st.info("Zona B: Vibración aceptable.")
    else:
        st.success("Zona A: Vibración excelente.")


# 2. Espacio para Observaciones del Ingeniero
st.write("---")
st.subheader("📝 Notas del Analista")
observaciones = st.text_area("Escribe aquí tus conclusiones adicionales para el PDF:", 
                             "Por ejemplo: Se observa que el aumento del espesor de la placa desplaza la frecuencia natural hacia arriba, reduciendo la amplitud en el punto de operación.")

st.info("💡 **Consejo para el reporte:** Las anotaciones de arriba aparecerán en tu PDF final.")

st.divider()
st.subheader("🖨️ Generar Reporte Técnico")

if st.button("Preparar Informe para PDF"):
    st.balloons()
    st.info("### Instrucciones para un PDF Profesional:\n"
            "1. Presiona **Ctrl + P** (Windows) o **Cmd + P** (Mac).\n"
            "2. Selecciona **'Guardar como PDF'**.\n"
            "3. En 'Más ajustes', activa **'Gráficos de fondo'**.\n"
            "4. Cambia el diseño a **'Vertical'**.")
    
    # Esto fuerza a Streamlit a mostrar todo de forma estática y clara
    st.markdown("""
        <style>
        @media print {
            .stButton, .stDownloadButton { display: none; } /* Oculta botones al imprimir */
            .main { background-color: white !important; }
        }
        </style>
    """, unsafe_allow_html=True)


st.sidebar.divider()
st.sidebar.header("💾 Gestión de Archivos")

# --- 1. SECCIÓN DE IMPORTAR (Cargar) ---
archivo_subido = st.sidebar.file_uploader("📂 Subir configuración (.json)", type=["json"])

if archivo_subido is not None:
    # Agregamos el botón para confirmar la carga
    if st.sidebar.button("📥 Aplicar configuración del archivo"):
        try:
            datos_preset = json.load(archivo_subido)
            
            # Actualizamos los componentes
            if "componentes_data" in datos_preset:
                for nombre, data in datos_preset["componentes_data"].items():
                    if nombre in st.session_state.componentes_data:
                        st.session_state.componentes_data[nombre].update(data)            
            
            if "configuracion_sistema" in datos_preset:
                st.session_state.configuracion_sistema.update(datos_preset["configuracion_sistema"])

            if "placa_data" in datos_preset:
                st.session_state.placa_data.update(datos_preset["placa_data"])

            # Actualización de Dampers
            if "dampers_prop_data" in datos_preset:
                st.session_state.dampers_prop_data = datos_preset["dampers_prop_data"]
                if "editor_tipos_nombres" in st.session_state:
                    del st.session_state["editor_tipos_nombres"]
            
            if "dampers_pos_data" in datos_preset:
                st.session_state.dampers_pos_data = datos_preset["dampers_pos_data"]
                if "pos_dampers_editor_v2" in st.session_state:
                    del st.session_state["pos_dampers_editor_v2"]


            for nombre in ["bancada", "cesto"]:
                for axis in ["x", "y", "z"]:
                    key = f"{axis}_{nombre}"
                    if key in st.session_state:
                        del st.session_state[key]

            st.sidebar.success("✅ Datos cargados correctamente")
            st.rerun() 
            
        except Exception as e:
            st.sidebar.error(f"Error al procesar el archivo: {e}")

# 3️⃣ GUARDADO ARCHIVO

def json_compacto(obj):
    """
    Convierte a JSON colapsando listas de números en una sola línea
    sin duplicar comas.
    """
    # 1. Generar JSON estándar
    content = json.dumps(obj, indent=4, sort_keys=True)
    
    # 2. Regex corregida: 
    # Busca una lista que empiece por '[', contenga números, comas, espacios y cierre con ']'
    # Luego elimina los saltos de línea y espacios extra dentro de esa lista.
    def limpiar_lista(match):
        return match.group(0).replace("\n", "").replace(" ", "").replace(",", ", ")

    # Esta regex identifica patrones de listas de números/floats
    content = re.sub(r'\[(?:\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*,?)+ \s*\]', limpiar_lista, content)
    
    # Limpieza final de seguridad por si quedaron espacios raros
    content = content.replace(", ]", "]").replace("[, ", "[")
    
    return content

# --- FUNCIONALIDAD DE EXPORTAR (Download) ---
# Preparamos el diccionario con todo lo que hay en memoria actualmente
datos_a_exportar = {
    # Agrupamos todo lo referente a la física global del sistema
    "configuracion_sistema": {
        "distancia_eje": st.session_state.configuracion_sistema["distancia_eje"],
        "diametro_cesto": st.session_state.configuracion_sistema["diametro_cesto"], 
        "sensor_pos": st.session_state.configuracion_sistema["sensor_pos"]
    },
    # Los diccionarios de componentes (Bancada, Cesto)
    "componentes_data": st.session_state.componentes_data,
    "placa": st.session_state.placa_data,    
    # Las dos tablas de los Dampers (Propiedades y Ubicaciones)
    "dampers_prop_data": st.session_state.dampers_prop_data,
    "dampers_pos_data": st.session_state.dampers_pos_data
}

# Convertir a string JSON
json_string = json_compacto(datos_a_exportar)
st.sidebar.download_button(
    label="📥 Descargar Configuración (.json)",
    data=json_string,
    file_name="config_centrifuga.json",
    mime="application/json",
    help="Guarda todos los datos actuales en un archivo para usarlos después."
)
st.sidebar.write("---")
