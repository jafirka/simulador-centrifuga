import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math
import copy
import json
import pandas as pd
import re


# --- INICIALIZACIÓN DE SESSION STATE ---
# Solo se ejecuta la primera vez que se abre la app
if 'componentes_data' not in st.session_state:
    st.session_state.componentes_data = {
        "bancada": {"m": 3542.0, "pos": [0.194, 0.0, 0.859], "I": [[3235.0, 0, 0], [0, 3690.0, 0], [0, 0, 2779.0]]},
        "motor": {"m": 940.0, "pos": [1.6, 0.0, 1.1], "I": [[178.0, 0, 0], [0, 392.0, 0], [0, 0, 312.0]]},
        "cesto": {"m": 1980.0, "pos": [0.5, 0.0, 0.0], "I": [[178.0, 0, 0], [0, 392.0, 0], [0, 0, 312.0]]}
    }

if 'dampers_prop_data' not in st.session_state:
    st.session_state.dampers_prop_data = [
        {"Tipo": "Ref_1", "kx": 1.32e6, "ky": 1.32e6, "kz": 1.6e6, "cx": 2.5e4, "cy": 2.5e4, "cz": 5e4},
        {"Tipo": "Ref_2", "kx": 1.0e6,  "ky": 1.0e6,  "kz": 1.3e6, "cx": 2.5e4, "cy": 2.5e4, "cz": 5e4}
    ]

if 'dampers_pos_data' not in st.session_state:
    st.session_state.dampers_pos_data = [
        {"Nombre": "D1 (Motor)", "X": 1.12, "Y": 0.84, "Z": 0.0, "Tipo": "Ref_1"},
        {"Nombre": "D2 (Motor)", "X": 1.12, "Y": -0.84, "Z": 0.0, "Tipo": "Ref_1"},
        {"Nombre": "D3 (Front)", "X": -0.93, "Y": 0.84, "Z": 0.0, "Tipo": "Ref_2"},
        {"Nombre": "D4 (Front)", "X": -0.93, "Y": -0.84, "Z": 0.0, "Tipo": "Ref_2"},
    ]

if 'placa_data' not in st.session_state:
    st.session_state.placa_data = {
        "lado_a": 2.4, "lado_b": 2.4, "espesor": 0.1, 
        "radio_agujero": 0.5, "dist_A": 0.0, "dist_B": 0.0
    }



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
            [1, 0, 0, 0,  lz, -ly],
            [0, 1, 0, -lz, 0,  lx],
            [0, 0, 1,  ly, -lx, 0]
        ])

    def get_matriz_K(self): return np.diag([self.kx, self.ky, self.kz])
    def get_matriz_C(self): return np.diag([self.cx, self.cy, self.cz])

class SimuladorCentrifuga:
    def __init__(self, config):
        self.pos_sensor = np.array(config["sensor"]["pos_sensor"])
        self.eje_vertical = config['eje_vertical'].lower()
        # --- Parámetros de la Placa ---
        p = config['placa']
        l_a, l_b, esp = p['lado_a'], p['lado_b'], p['espesor']
        dist_A, dist_B = p.get('dist_A', 0), p.get('dist_B', 0)
        r = p['radio_agujero']
        rho = 7850 # kg/m^3 (Acero)


	# --- Mapeo dinámico de dimensiones a ejes X, Y, Z ---
        # El espesor siempre va en el eje vertical. 
        # Los lados a y b se reparten en los ejes restantes.
        if self.eje_vertical == 'z':
            dx, dy, dz = l_a, l_b, esp
            pos_placa = [dist_A, dist_B, 0]
        elif self.eje_vertical == 'y':
            dx, dy, dz = l_a, esp, l_b
            pos_placa = [dist_A, 0, dist_B]
        else: # eje x
            dx, dy, dz = esp, l_a, l_b
            pos_placa = [0, dist_A, dist_B]
        self.dims = {'x': dx, 'y': dy, 'z': dz}

        # 1. Masa: Masa total - Masa del agujero
        m_total = (dx * dy * dz) * rho
        m_agujero = (math.pi * r**2 * esp) * rho
        self.m_placa = m_total - m_agujero

        # 2. Inercias respecto al CG de la placa
        I_final = {}
        for eje in ['x', 'y', 'z']:
            # Dimensiones perpendiculares al eje de cálculo
            d_perp = [v for k, v in self.dims.items() if k != eje]
            d1, d2 = d_perp
            
            # Inercia Bloque sólido
            I_b = (1/12) * m_total * (d1**2 + d2**2)
            
            # Inercia Agujero (Cilindro)
            if eje == self.eje_vertical:
                I_a = (1/2) * m_agujero * r**2 # Eje polar
            else:
                I_a = (1/4) * m_agujero * (r**2 + (esp**2)/3) # Eje diametral

            I_final[eje] = I_b - I_a

        self.I_placa = [I_final['x'], I_final['y'], I_final['z']]




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
        
        for c in self.componentes.values():
            m_c = c["m"]
            p_c = np.array(c["pos"])
            # Inercia local (convertir a matriz 3x3 si es lista)
            I_local = np.diag(c["I"]) if isinstance(c["I"], list) else np.array(c["I"])
        
            # Vector desde el CG global al CG del componente
            d = p_c - cg_global
        
            # Teorema de Steiner (Ejes Paralelos) en forma matricial
            # I_global = sum( I_local + m * [ (d·d)diag(1) - (d ⊗ d) ] )
            term_steiner = m_c * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            I_global += (I_local + term_steiner)

        M[0:3, 0:3], M[3:6, 3:6] = np.eye(3) * m_total, I_global
        K, C = np.zeros((6, 6)), np.zeros((6, 6))
        for damper in self.dampers:
            T = damper.get_matriz_T(cg_global)
            K += T.T @ damper.get_matriz_K() @ T
            C += T.T @ damper.get_matriz_C() @ T

        return M, K, C, cg_global

    def calcular_frecuencias_naturales(self):
      M, K, C, _ = self.armar_matrices()
    
      # Resolvemos el problema de autovalores generalizado: K * v = lambda * M * v
      # evals son los autovalores (w^2), evecs son los modos de vibración
      evals, evecs = linalg.eigh(K, M)
    
      # evals pueden ser negativos muy pequeños por precisión numérica, los limpiamos
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
# [Aquí pegas la función ejecutar_barrido_rpm de los bloques 12 a 19] [cite: 12, 19]

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

        # Construcción del vector de excitación F (6 componentes)
        if eje_vertical == 'x':
            arm = dist - cg_global[0]
            # Fuerza en Y y Z | Momento en Y (debido a Fz) y Momento en Z (debido a Fy)
            F = np.array([0, F0, F0*1j, 0, -(F0*1j)*arm, F0*arm])
        elif eje_vertical == 'y':
            arm = dist - cg_global[1]
            # Fuerza en X y Z | Momento en X (debido a Fz) y Momento en Z (debido a Fx)
            F = np.array([F0, 0, F0*1j, (F0*1j)*arm, 0, -F0*arm])
            #F = np.array([F0*1j, 0, F0, (F0)*arm, 0, -(F0*1j)*arm])
        else: # Eje Z
            arm = dist - cg_global[2]
            # Fuerza en X y Y | Momento en X (debido a Fy) y Momento en Y (debido a Fx)
            F = np.array([F0, F0*1j, 0, (F0*1j)*arm, -F0*arm, 0])

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
    
    return rpm_range, D_desp, D_fuerza, acel_cg, vel_cg, S_desp, S_vel, S_acel



# ==========================================
# 3️⃣ ENTORNO VISUAL (INTERFAZ)
# ==========================================
import copy
# --- 3. INTERFAZ DE STREAMLIT ---
st.set_page_config(layout="wide")
st.title("Simulador Interactivo de Centrífuga 300F - Departamento de Ingenieria de Riera Nadeu")
st.markdown("Modifica los valores en la barra lateral para ver el impacto en las vibraciones.")

# --- BARRA LATERAL PARA MODIFICAR VALORES ---
st.sidebar.header("Parámetros de cálculos")

# Ejemplo de cómo modificar la masa de desbalanceo y RPM
m_unbalance = st.sidebar.slider("Masa de Desbalanceo (kg)", 0.1, 8.0, 1.6)
rpm_obj = st.sidebar.number_input("RPM nominales", value=1100)




# --- SECCIÓN: GESTIÓN DE COMPONENTES DINÁMICOS ---
st.header("🧱 Configuración del Sistema")

# Contenedor para los datos procesados en los tabs
comp_editados = {} 

tab_comp, tab_dampers, tab_config = st.tabs(["📦 Componentes Masas/Inercias", "🛡️ Configuración de Dampers", "⚙️ Configuración del Sistema"])

with tab_config:
    st.subheader("Configuración de Ejes y Convención")
    col_sys1, col_sys2 = st.columns(2)
    
    with col_sys1:
        # --- Definir ejes de referencia ---

        eje_vertical = st.selectbox(
        "Eje de Rotación (Vertical)", 
        ('x', 'y', 'z'), 
        index=('x', 'y', 'z').index(st.session_state.get('eje_vertical_val', 'z'))
        )

        # 2. Distancia de la masa de desbalanceo
        distancia_eje = st.number_input(
            "Coordenada vertical de la masa de desbalanceo (m)", 
            value=float(st.session_state.get('distancia_eje_val', 0.8)),
            key="sys_dist_eje"
        )

        # Determinar el plano del rotor en función del eje vertical
        if eje_vertical == 'x':
            plano_rotor = ['y', 'z']
        elif eje_vertical == 'y':
            plano_rotor = ['z', 'x']
        else: # 'z'
            plano_rotor = ['x', 'y']
       
        # --- NUEVA SECCIÓN: POSICIÓN DEL SENSOR ---
        st.text("Posición del Sensor de velocidad/aceleracion(m)")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sensor_x = st.number_input("X", value=float(st.session_state.get("sensor_x_val", 0.0)), key="s_x")
        with col_s2:
            sensor_y = st.number_input("Y", value=float(st.session_state.get("sensor_y_val", 0.8)), key="s_y")
        with col_s3:
            sensor_z = st.number_input("Z", value=float(st.session_state.get("sensor_z_val", 0.0)), key="s_z")

    with col_sys2:
        # Un resumen rápido de los valores globales para no tener que buscarlos en el sidebar
        st.markdown(f"**Material:** Acero ($\rho$ = 7850 kg/m³)")

    # Actualizamos los valores de sistema en el session_state con lo que hay actualmente en los widgets
    st.session_state["eje_vertical_val"] = eje_vertical
    st.session_state["distancia_eje_val"] = distancia_eje
    st.session_state["sensor_pos_val"] = [sensor_x, sensor_y, sensor_z] 

    st.divider()




# 1️⃣ GESTIÓN DE COMPONENTES (Inercia 3x3 con Persistencia)
with tab_comp:
    subtabs = st.tabs(["Bancada", "Accionamiento", "Cesto", "Placa inercia"])
    
    # Mapeo de nombres para session_state
    nombres_llaves = ["bancada", "motor", "cesto"]

    for i, nombre in enumerate(nombres_llaves):
        with subtabs[i]:
            # ✅ NOTA ACLARATORIA: Solo para la primera subpestaña (Bancada)
            if i == 0:
                st.info("💡 **Nota:** La bancada debe inlcuir la masa y la inercia de la caja de rodamientos y elementos auxilaires, EXCLUYENDO la placa de inercia")
            # ✅ RECUPERAR DATOS: Priorizamos la memoria de la sesión
            datos_memoria = st.session_state.componentes_data[nombre]
        
            
            c_m, c_p = st.columns([1, 2])
            with c_m:
                m_val = st.number_input(
                    f"Masa {nombre} (kg)", 
                    value=float(st.session_state.componentes_data[nombre].get("m", 0.0)), 
                    key=f"m_input_{nombre}" # Esta key se limpia al importar
                )
            with c_p:
                cx, cy, cz = st.columns(3)
                # Leemos las posiciones del array 'pos' del JSON
                pos_actual = st.session_state.componentes_data[nombre].get("pos", [0.0, 0.0, 0.0])
                px = cx.number_input(f"X [m]", value=float(pos_actual[0]), key=f"px_in_{nombre}")
                py = cy.number_input(f"Y [m]", value=float(pos_actual[1]), key=f"py_in_{nombre}")
                pz = cz.number_input(f"Z [m]", value=float(pos_actual[2]), key=f"pz_in_{nombre}")

            st.write(f"**Matriz de Inercia (3x3) para {nombre} [kg·m²]**")
            
            # ✅ EDITOR DE MATRIZ: Carga lo que hay en memoria (I)
            # Convertimos a array de numpy por seguridad
            iner_inicial = np.array(datos_memoria["I"])
            
            df_iner_3x3 = st.data_editor(
                iner_inicial,
                key=f"matrix_3x3_{nombre}",
                use_container_width=True,
                column_config={"0": "X", "1": "Y", "2": "Z"}
            )
            
            # ✅ ACTUALIZAR MEMORIA: Guardamos los cambios inmediatamente
            st.session_state.componentes_data[nombre] = {
                "m": m_val, 
                "pos": [px, py, pz], 
                "I": df_iner_3x3.tolist() if isinstance(df_iner_3x3, np.ndarray) else df_iner_3x3
            }

            # Esto es lo que usará config_base al final del script
            comp_editados[nombre] = st.session_state.componentes_data[nombre]


# 2. ✅ NUEVA SUBTAB: Datos de la Placa de Inercia
with subtabs[3]:
    st.write("### Parámetros Geométricos de la Placa")
    
    # Creamos las columnas primero para organizar la interfaz
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Usamos el valor guardado en session_state como 'value'

        lado_a = st.number_input("Lado A [m]", 
                                value=float(st.session_state.placa_data.get("lado_a", 2.4)), 
                                step=0.1, format="%.2f", key="input_lado_a")
        
        lado_b = st.number_input("Lado B [m]", 
                                value=float(st.session_state.placa_data.get("lado_b", 2.4)), 
                                step=0.1, format="%.2f", key="input_lado_b")
    
    with col_g2:
        espesor = st.number_input("Espesor [m]", 
                                    value=float(st.session_state.placa_data.get("espesor", 0.1)), 
                                    step=0.01, format="%.3f", key="input_espesor")
        radio_agujero = st.number_input("Radio Agujero [m]", 
                                        value=float(st.session_state.placa_data.get("radio_agujero", 0.5)), 
                                        step=0.05, format="%.2f", key="input_radio")

    st.write("### Posición del Centro de la Placa")
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        dist_A = st.number_input(f"Desfase en {plano_rotor[0].upper()} (dist_A)", 
                                 value=st.session_state.placa_data["dist_A"], 
                                 step=0.1, key="input_dist_A")
    with col_p2:
        dist_B = st.number_input(f"Desfase en {plano_rotor[1].upper()} (dist_B)", 
                                 value=st.session_state.placa_data["dist_B"], 
                                 step=0.1, key="input_dist_B")

    # ✅ ACTUALIZACIÓN CRÍTICA: Guardamos los valores actuales en el session_state
    # Esto permite que al presionar "Descargar JSON", los datos estén al día.
    st.session_state.placa_data.update({
        "lado_a": lado_a,
        "lado_b": lado_b,
        "espesor": espesor,
        "radio_agujero": radio_agujero,
        "dist_A": dist_A,
        "dist_B": dist_B
    })


# 2️⃣ GESTIÓN DE DAMPERS
dampers_finales = []
with tab_dampers:
    st.write("### 1. Definición de Propiedades por Tipo")
    
    # 1. Editor de Propiedades (Ya lo tenías bien, pero aseguramos el guardado)
    df_prop_editada = st.data_editor(
        st.session_state.dampers_prop_data,
        key="editor_tipos_nombres",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tipo": st.column_config.TextColumn("Nombre del Tipo/Modelo", required=True),
            "kx": st.column_config.NumberColumn("Kx [N/m]"),
            "ky": st.column_config.NumberColumn("Ky [N/m]"),
            "kz": st.column_config.NumberColumn("Kz [N/m]"),
        }
    )
    # IMPORTANTE: Guardar el cambio en el estado para que el JSON sepa qué hay
    st.session_state.dampers_prop_data = df_prop_editada

    df_prop = pd.DataFrame(df_prop_editada)
    lista_tipos_disponibles = df_prop["Tipo"].tolist()

    st.write("### 2. Ubicación de los Dampers")
    
    # 2. Editor de Posiciones (CORREGIDO para usar Session State)
    res_pos_editor = st.data_editor(
        st.session_state.dampers_pos_data, # <--- USAR ESTADO, NO LA LISTA FIJA
        num_rows="dynamic", 
        key="pos_dampers_editor_v2", 
        use_container_width=True,
        column_config={
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo de Damper", 
                options=lista_tipos_disponibles,
                required=True
            )
        }
    )
    # IMPORTANTE: Guardar el cambio en el estado
    st.session_state.dampers_pos_data = res_pos_editor

    # ✅ PROCESAMIENTO FINAL
    df_pos = pd.DataFrame(res_pos_editor)
    df_prop_indexed = df_prop.set_index("Tipo")

    for _, row in df_pos.iterrows():
        tipo_sel = row.get("Tipo") # Usamos .get por seguridad si la fila está vacía
        if tipo_sel in df_prop_indexed.index:
            p = df_prop_indexed.loc[tipo_sel]
            dampers_finales.append({
                "nombre": row["Nombre"],
                "pos": [row["X"], row["Y"], row["Z"]],
                "tipo": tipo_sel,
                "kx": p["kx"], "ky": p["ky"], "kz": p["kz"],
                "cx": p["cx"], "cy": p["cy"], "cz": p["cz"]
            })


            
# 3️⃣ ENSAMBLAJE FINAL
config_base = {
    "eje_vertical": eje_vertical, 
    "plano_rotor": plano_rotor,
    "excitacion": {"distancia_eje": distancia_eje, "m_unbalance": m_unbalance, "e_unbalance": 0.8},
    "placa": {"lado_a": lado_a, "lado_b": lado_b, "espesor": espesor, "radio_agujero": radio_agujero, "dist_A": dist_A, "dist_B": dist_B},
    "componentes": comp_editados,
    "dampers": dampers_finales,
    "sensor": {"pos_sensor": [sensor_x, sensor_y, sensor_z]},
    "tipos_dampers": df_prop.to_dict('index')

}

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

st.sidebar.divider()
st.sidebar.header("💾 Gestión de Archivos")
# --- FUNCIONALIDAD DE EXPORTAR (Download) ---
# Preparamos el diccionario con todo lo que hay en memoria actualmente
datos_a_exportar = {
    "configuracion_sistema": {
        "eje_vertical": st.session_state["eje_vertical_val"],
        "distancia_eje": st.session_state["distancia_eje_val"],
        "sensor_pos": st.session_state["sensor_pos_val"]
    },
    "componentes_data": st.session_state.componentes_data,
    "dampers_prop_data": st.session_state.dampers_prop_data,
    "dampers_pos_data": st.session_state.dampers_pos_data,
    "placa_data": st.session_state.placa_data
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



# --- FUNCIONALIDAD DE IMPORTAR (Generalizada) ---
archivo_subido = st.sidebar.file_uploader("📂 Cargar archivo JSON", type=["json"])

if archivo_subido is not None:
    try:
        datos_cargados = json.load(archivo_subido)
        
        if st.sidebar.button("🚀 Aplicar Configuración Completa"):
            # --- PASO 1: Limpiar TODO el session_state ---
            for key in list(st.session_state.keys()):
                # Opcional: mantener el archivo subido para que no desaparezca el widget
                if key != "archivo_subido_key": 
                    del st.session_state[key]
            
            # --- PASO 2: Inyectar los datos del JSON al session_state ---
            # Esto repuebla los diccionarios base
            for key, value in datos_cargados.items():
                st.session_state[key] = value
            
            # --- PASO 3: Sincronizar variables de sistema (Ejes, RPM, etc.) ---
            # Si guardaste 'configuracion_sistema', creamos las llaves individuales
            if "configuracion_sistema" in datos_cargados:
                cfg = datos_cargados["configuracion_sistema"]
                st.session_state["eje_vertical_val"] = cfg.get("eje_vertical", "z")
                st.session_state["distancia_eje_val"] = cfg.get("distancia_eje", 0.8)
                
                # Sincronizar también las posiciones del sensor
                s_pos = cfg.get("sensor_pos", [0.0, 0.8, 0.0])
                st.session_state["sensor_x_val"], st.session_state["sensor_y_val"], st.session_state["sensor_z_val"] = s_pos

            st.sidebar.success("✅ ¡Sistema reseteado y cargado!")
            st.rerun() 
            
    except Exception as e:
        st.sidebar.error(f"Error al procesar el archivo: {e}")



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


# Ejes

vertical = config_base["eje_vertical"]
horizontales = config_base["plano_rotor"]  # lista con los dos ejes del plano horizontal

# Colores y etiquetas
colores = {horizontales[0]: "tab:blue", horizontales[1]: "tab:orange", vertical: "tab:green"}
ejes_lbl = {horizontales[0]: horizontales[0].upper(),
            horizontales[1]: horizontales[1].upper(),
            vertical: vertical.upper()}

ejes_lbl = {horizontales[0]: "Horizontal 1", horizontales[1]: "Horizontal 2", vertical: "Vertical"}


# RPM de operación

rpm_range = np.linspace(10, rpm_obj*1.2, 1000)
idx_op = np.argmin(np.abs(rpm_range - rpm_obj))

rpm_range, D_desp, D_fuerza, acel_cg, vel_cg, S_desp, S_vel, S_acel = ejecutar_barrido_rpm(modelo_base, rpm_range, d_idx)
rpm_range, desp_prop, fuerza_prop, acel_prop, vel_prop, S_desp_prop, S_vel_prop, S_acel_prop = ejecutar_barrido_rpm(modelo_prop, rpm_range, d_idx)

# ==========================================
# 📄 INTRODUCCIÓN Y MEMORIA DE CÁLCULO
# ==========================================
st.markdown(f"""
### 📋 Resultados
---
""")

# --- Visualización del Layout de la Máquina ---
col_map1, col_map2 = st.columns([1, 1])

with col_map1:
    st.write(f"**Mapa de Ubicación (Plano perpendicular al eje {eje_vertical.upper()})**")
    fig_map, ax_map = plt.subplots(figsize=(5, 5))

    # --- 1. Determinar qué ejes graficar según el eje vertical ---
    # Si vertical es Z -> Graficamos X e Y
    # Si vertical es Y -> Graficamos X e Z
    # Si vertical es X -> Graficamos Y e Z
    if eje_vertical == 'z':
        idx_h, idx_v = 0, 1  # Horizontal=X, Vertical=Y
        label_h, label_v = "X [m]", "Y [m]"
    elif eje_vertical == 'y':
        idx_h, idx_v = 0, 2  # Horizontal=X, Vertical=Z
        label_h, label_v = "X [m]", "Z [m]"
    else: # 'x'
        idx_h, idx_v = 1, 2  # Horizontal=Y, Vertical=Z
        label_h, label_v = "Y [m]", "Z [m]"

# Extraemos la posición del componente placa definido en el simulador
    pos_placa_completa = modelo_base.componentes["placa"]["pos"] 
    pos_h_placa = pos_placa_completa[idx_h]
    pos_v_placa = pos_placa_completa[idx_v]

    lado_a = config_base["placa"]["lado_a"]
    lado_b = config_base["placa"]["lado_b"]
    
# ✅ MODIFICADO: El anclaje ahora depende de la posición real del CG de la placa
    anclaje_h = pos_h_placa - lado_a / 2
    anclaje_v = pos_v_placa - lado_b / 2

    rect = plt.Rectangle((anclaje_h, anclaje_v), lado_a, lado_b, 
                         color='lightgray', alpha=0.3, label='Placa Base')
    ax_map.add_patch(rect)
    
    # Dibujar dampers
    for i, d in enumerate(modelo_base.dampers):
        es_seleccionado = (i == d_idx)
        ax_map.scatter(d.pos[0], d.pos[1], 
                       c='red' if es_seleccionado else 'blue', 
                       s=200, zorder=3, 
                       label='Analizado' if es_seleccionado else None)
        ax_map.text(d.pos[0]+0.1, d.pos[1]+0.1, f"D{i}", fontsize=12, fontweight='bold')

    # 3. Dibujar posición del MOTOR BASE (Gris/Referencia)
    pos_motor_base = config_base["componentes"]["motor"]["pos"]
    ax_map.scatter(pos_motor_base[0], pos_motor_base[1], 
                   marker='s', s=150, color='gray', alpha=0.5, label='Motor Base', zorder=4)

    # 4. Dibujar posición del MOTOR PROPUESTA (Verde/Activo)
    # Usamos el valor del slider directamente
    ax_map.scatter(pos_x_motor_prop, 0, 
            marker='s', s=180, color='green', edgecolors='black', label='Motor Propuesta', zorder=5)    

    ax_map.axhline(0, color='black', lw=1); ax_map.axvline(0, color='black', lw=1)
    ax_map.axvline(0, color='black', lw=0.8, ls='--')
    ax_map.set_xlim(-2, 2); ax_map.set_ylim(-2, 2)
    ax_map.set_xlabel(f"{plano_rotor[0].upper()} [m]")
    ax_map.set_ylabel(f"{plano_rotor[1].upper()} [m]")
    ax_map.grid(True, alpha=0.2)
    st.pyplot(fig_map)

with col_map2:
    st.write("**Distribución de Masas (Centro de Gravedad)**")
    # Calcular CG para mostrarlo
    M_sys, _, _, cg_final = modelo_base.armar_matrices()
    esp_base_mm = config_base["placa"]["espesor"] * 1000  # Convertir a mm
    st.info(f"""
    **Centro de Gravedad Global (Base):**
    * X: {cg_final[0]:.3f} m
    * Y: {cg_final[1]:.3f} m
    * Z: {cg_final[2]:.3f} m
    
    *Espesor de placa base: {esp_base_mm:.1f} mm*
    """)

    # Calcular CG para mostrarlo
    M_sys, _, _, cg_final = modelo_prop.armar_matrices()
    esp_prop_mm = config_prop["placa"]["espesor"] * 1000  # Convertir a mm
    st.info(f"""
    **Centro de Gravedad Global (propuesto):**
    * X: {cg_final[0]:.3f} m
    * Y: {cg_final[1]:.3f} m
    * Z: {cg_final[2]:.3f} m
    
    *Espesor de placa base: {esp_prop_mm:.1f} mm*
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

# ==========================
# 📊 GRÁFICO 1: Aceleracion en el SENSOR
# ==========================


st.subheader("Análisis de Aceleración en el Sensor")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rpm_range, S_acel["x"], label="Horizontal X")
ax.plot(rpm_range, S_acel["y"], label="Horizontal Y")
ax.plot(rpm_range, S_acel["z"], label="Vertical Z")
ax.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax.set_xlabel("RPM")
ax.set_ylabel("Aceleración [g]")
ax.grid(True, alpha=0.1)
ax.legend()
plt.rcParams.update({'font.size': 10}) # Fuente más legible para impresión
fig.tight_layout() # Evita que las etiquetas se corten
st.pyplot(fig, clear_figure=True)


# ==========================
# 📊 GRÁFICO 2: Velocidad en el SENSOR
# ==========================
st.subheader("Respuesta en Frecuencia: Velocidad en Sensor")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for eje in [horizontales[0], horizontales[1], eje_vertical]:
    ax2.plot(rpm_range, S_vel[eje], color=colores[eje], label=f'{ejes_lbl[eje]}')
ax2.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
for f in f_res_rpm:
    if f < rpm_range[-1]: # Solo si está dentro del rango del gráfico
        plt.axvline(f, color='red', linestyle='--', alpha=0.5, label='Resonancia Teórica' if f == f_res_rpm[0] else "")
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
for eje in [horizontales[0], horizontales[1], eje_vertical]:
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
for eje in [horizontales[0], horizontales[1], eje_vertical]:
    ax4.plot(rpm_range, D_fuerza[eje], color=colores[eje], label=f'{ejes_lbl[eje]}')
ax4.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')

# Anotación de fuerza a RPM operación (usando el eje vertical)
f_max_op = D_fuerza[eje_vertical][idx_op]
ax4.annotate(
    f'{f_max_op:.0f} N a {rpm_obj} RPM',
    xy=(rpm_range[idx_op], f_max_op),
    xytext=(rpm_range[idx_op] * 0.7, f_max_op * 1.05),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5)
)
ax4.set_xlabel('Velocidad de Rotación [RPM]')
ax4.set_ylabel('Fuerza [N]')
ax4.grid(True, alpha=0.1)
ax4.legend()
st.pyplot(fig4)

# Inserta esto antes de una sección nueva que quieras que empiece en hoja limpia
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.title("Simulador de variaciones de parametros")
st.subheader("Comparativa de velocidad en el Sensor")

fig, ax = plt.subplots(figsize=(10, 4))
# --- CASO BASE (Líneas punteadas o grises) ---
ax.plot(rpm_range, S_vel["x"], color="gray", linestyle="--", alpha=0.5, label="Base X")
ax.plot(rpm_range, S_vel["y"], color="silver", linestyle="--", alpha=0.5, label="Base Y")
ax.plot(rpm_range, S_vel["z"], color="black", linestyle="--", alpha=0.5, label="Base Z")
# --- PROPUESTA (Colores vivos) ---
ax.plot(rpm_range, S_vel_prop["x"], color="tab:blue", label="Propuesta X")
ax.plot(rpm_range, S_vel_prop["y"], color="tab:orange", label="Propuesta Y")
ax.plot(rpm_range, S_vel_prop["z"], color="tab:green", label="Propuesta Z")
ax.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax.set_xlabel("RPM")
ax.set_ylabel("velocidad [mm/s]")
ax.legend()
ax.grid(True, alpha=0.1)
st.pyplot(fig)

st.subheader(f"Comparativa de fuerza vertical en el Damper {lista_dampers_config[d_idx]['tipo']}")

fig, ax = plt.subplots(figsize=(10, 4))
# --- CASO BASE (Líneas punteadas o grises) ---
ax.plot(rpm_range, D_fuerza[vertical], color="gray", linestyle="--", alpha=0.5, label="Base X")
# --- PROPUESTA (Colores vivos) ---
ax.plot(rpm_range, fuerza_prop[vertical], color="tab:blue", label="Propuesta X")
Fz_orig_1100 = D_fuerza[vertical][idx_op]
Fz_prop_1100 = fuerza_prop[vertical][idx_op]
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
ax.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax.set_xlabel("RPM")
ax.set_ylabel("Fuerza [N]")
ax.legend()
ax.grid(True, alpha=0.1)
st.pyplot(fig)



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
idx_res_base = np.argmax(S_vel[vertical])
rpm_res_base = rpm_range[idx_res_base]

idx_res_prop = np.argmax(S_vel_prop[vertical])
rpm_res_prop = rpm_range[idx_res_prop]

col_concl1, col_concl2 = st.columns(2)

with col_concl1:
    st.write("### 🚨 Puntos Críticos (Resonancia)")
    # Mostramos la primera frecuencia natural (Modo 1)
    st.write(f"**Caso Base (Modo 1):** {f_res_rpm[0]:.0f} RPM")
    st.write(f"**Caso Base (Modo 6):** {f_res_rpm[5]:.0f} RPM")
    st.write(f"**Propuesta (Modo 1):** {f_res_rpm_prop[0]:.0f} RPM")
    st.write(f"**Propuesta (Modo 6):** {f_res_rpm_prop[5]:.0f} RPM")
    
    dist_min_base = abs(f_res_rpm[5] - rpm_obj)
    dist_min_prop = np.min(np.abs(f_res_rpm_prop[5] - rpm_obj))

    if dist_min_base < 150 or dist_min_prop < 150:
        # Identificamos cuál falló para dar un mensaje preciso
        fallo_en = "la Propuesta" if dist_min_prop < 150 else "el Caso Base"
        if dist_min_base < 150 and dist_min_prop < 150:
            fallo_en = "Ambos Modelos"
            
        st.error(f"⚠️ PELIGRO: Resonancia crítica detectada en {fallo_en}. "
                 f"Margen insuficiente (< 150 RPM) respecto a {rpm_obj} RPM.")
    else:
        st.success(f"✅ SEGURO: Todos los modos de ambos modelos mantienen un margen "
                   f"> 150 RPM respecto a la operación.")
        
    st.caption(f"Margen actual: Base {dist_min_base:.0f} RPM | Propuesta {dist_min_prop:.0f} RPM")

with col_concl2:
    st.write("### 📊 Cumplimiento de Norma (ISO 10816)")
    
    # Extraemos los picos máximos de velocidad
    v_max_base = max(max(S_vel["x"]), max(S_vel["y"]), max(S_vel["z"]))
    v_max_prop = max(max(S_vel_prop["x"]), max(S_vel_prop["y"]), max(S_vel_prop["z"]))
        
    # Mostramos ambos valores para el reporte
    st.write(f"**Velocidad Máx. Base:** {v_max_base:.2f} mm/s")
    st.write(f"**Velocidad Máx Propuesta:** {v_max_prop:.2f} mm/s")
    
    # Evaluación de severidad para la Propuesta
    # Definimos los umbrales de la norma
    if v_max_prop < 4.5:
        st.success(f"**Clase A/B:** {v_max_prop:.2f} mm/s (Excelente/Satisfactorio)")
        st.caption("La máquina puede operar de forma continua sin restricciones.")
    elif v_max_prop < 11.2:
        st.warning(f"**Clase C:** {v_max_prop:.2f} mm/s (Alerta/Insatisfactorio)")
        st.caption("No apta para operación continua a largo plazo. Requiere mantenimiento.")
    else:
        st.error(f"**Clase D:** {v_max_prop:.2f} mm/s (Peligro)")
        st.caption("Riesgo inminente de daño estructural. Detener operación.")

    # Cálculo de la mejora real
    mejora_vel = ((v_max_base - v_max_prop) / v_max_base) * 100
    if mejora_vel > 0:
        st.write(f"📈 Reducción de vibración: **{mejora_vel:.1f}%**")
    else:
        st.write(f"📉 Incremento de vibración: **{abs(mejora_vel):.1f}%**")



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
