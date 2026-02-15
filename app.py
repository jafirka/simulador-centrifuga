import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math
import copy

# ==========================================#
# 1️⃣ CLASES DEL MODELO
# ==========================================#

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
        if self.eje_vertical == 'z':
            dx, dy, dz = l_a, l_b, esp
        elif self.eje_vertical == 'y':
            dx, dy, dz = l_a, esp, l_b
        else: # eje x
            dx, dy, dz = esp, l_a, l_b
        self.dims = {'x': dx, 'y': dy, 'z': dz}

        # 1. Masa de la placa
        m_total = (dx * dy * dz) * rho
        m_agujero = (math.pi * r**2 * esp) * rho
        self.m_placa = m_total - m_agujero

        # 2. Inercias de la placa respecto a su propio CG
        I_final = {}
        for eje in ['x', 'y', 'z']:
            d_perp = [v for k, v in self.dims.items() if k != eje]
            d1, d2 = d_perp
            
            I_b = (1/12) * m_total * (d1**2 + d2**2)
            
            if eje == self.eje_vertical:
                I_a = (1/2) * m_agujero * r**2
            else:
                I_a = (1/4) * m_agujero * (r**2 + (esp**2)/3)
            I_final[eje] = I_b - I_a
        self.I_placa = [I_final['x'], I_final['y'], I_final['z']]

        # --- Posición dinámica de la placa ---
        if self.eje_vertical == 'z':
            pos_placa = [dist_A, dist_B, 0]
        elif self.eje_vertical == 'y':
            pos_placa = [dist_A, 0, dist_B]
        else: # eje x
            pos_placa = [0, dist_A, dist_B]

        # --- Componentes ---
        self.componentes = {
            "placa": {"m": self.m_placa, "pos": pos_placa, "I": self.I_placa},
            "cesto": config['componentes']['cesto'],
            "bancada": config['componentes']['bancada'],
            "motor": config['componentes']['motor']
        }
        
        # OPTIMIZACIÓN: Convertir inercias de la config a np.arrays
        # Esto permite que el diccionario de config sea "hasheable" para el caché
        for comp_dict in self.componentes.values():
            if "I" in comp_dict and isinstance(comp_dict["I"], list):
                comp_dict["I"] = np.array(comp_dict["I"])

        # --- Excitación y Dampers ---
        self.excitacion = config['excitacion']
        self.dampers = []
        for d_conf in config['dampers']:
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
            I_local = np.diag(c["I"]) if len(np.array(c["I"]).shape) == 1 else c["I"]
            d = np.array(c["pos"]) - cg_global
            # Teorema de Steiner
            I_st = I_local + c["m"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            I_global += I_st

        M[0:3, 0:3], M[3:6, 3:6] = np.eye(3) * m_total, I_global
        K, C = np.zeros((6, 6)), np.zeros((6, 6))
        for damper in self.dampers:
            T = damper.get_matriz_T(cg_global)
            K += T.T @ damper.get_matriz_K() @ T
            C += T.T @ damper.get_matriz_C() @ T

        return M, K, C, cg_global

    def calcular_frecuencias_naturales(self):
        M, K, C, _ = self.armar_matrices()
        evals, evecs = linalg.eigh(K, M)
        evals = np.maximum(evals, 0)
        w_n = np.sqrt(evals)
        f_rpm = (w_n / (2 * np.pi)) * 60
        return f_rpm, evecs

# ==========================================#
# 2️⃣ LÓGICA DE CÁLCULO
# ==========================================#

def ejecutar_barrido_rpm(modelo, rpm_range, d_idx):
    M, K, C, cg_global = modelo.armar_matrices()
    T_sensor = modelo.obtener_matriz_sensor(cg_global)

    damper_d = modelo.dampers[d_idx]
    T_damper = damper_d.get_matriz_T(cg_global)
    ks = [damper_d.kx, damper_d.ky, damper_d.kz]
    cs = [damper_d.cx, damper_d.cy, damper_d.cz]

    ex = modelo.excitacion
    dist = ex['distancia_eje']

    acel_cg, vel_cg = {"x": [], "y": [], "z": []}, {"x": [], "y": [], "z": []}
    D_desp, D_fuerza = {"x": [], "y": [], "z": []}, {"x": [], "y": [], "z": []}
    S_desp, S_vel, S_acel = {"x": [], "y": [], "z": []}, {"x": [], "y": [], "z": []}, {"x": [], "y": [], "z": []}

    for rpm in rpm_range:
        w = rpm * 2 * np.pi / 60
        F0 = ex['m_unbalance'] * ex['e_unbalance'] * w**2

        arm = dist - cg_global[{"x": 0, "y": 1, "z": 2}[modelo.eje_vertical]]
        if modelo.eje_vertical == 'x':
            F = np.array([0, F0, F0*1j, 0, -(F0*1j)*arm, F0*arm])
        elif modelo.eje_vertical == 'y':
            F = np.array([F0, 0, F0*1j, (F0*1j)*arm, 0, -F0*arm])
        else: # 'z'
            F = np.array([F0, F0*1j, 0, (F0*1j)*arm, -F0*arm, 0])

        Z = -w**2 * M + 1j*w * C + K
        X = linalg.solve(Z, F)
        
        X_damper = T_damper @ X
        U_sensor = T_sensor @ X

        for i, eje in enumerate(["x", "y", "z"]):
            acel_cg[eje].append((w**2) * np.abs(X[i]) / 9.81)
            vel_cg[eje].append(w * np.abs(X[i]) * 1000)
            D_desp[eje].append(np.abs(X_damper[i]) * 1000)
            D_fuerza[eje].append(np.abs((ks[i] + 1j * w * cs[i]) * X_damper[i]))
            S_desp[eje].append(np.abs(U_sensor[i]) * 1000)
            S_vel[eje].append(w * np.abs(U_sensor[i]) * 1000)
            S_acel[eje].append((w**2) * np.abs(U_sensor[i]) / 9.81)
    
    # REFACTORIZACIÓN: Devolver un diccionario en lugar de una tupla larga
    return {
        "rpm_range": rpm_range,
        "damper_desplazamiento": D_desp,
        "damper_fuerza": D_fuerza,
        "cg_aceleracion": acel_cg,
        "cg_velocidad": vel_cg,
        "sensor_desplazamiento": S_desp,
        "sensor_velocidad": S_vel,
        "sensor_aceleracion": S_acel
    }

# ==========================================#
# 3️⃣ FUNCIONES AUXILIARES Y DE CACHÉ
# ==========================================#

# OPTIMIZACIÓN: Cachear los resultados para evitar recalcular
@st.cache_data
def cached_natural_frequencies(config):
    modelo = SimuladorCentrifuga(config)
    return modelo.calcular_frecuencias_naturales()

@st.cache_data
def cached_run_rpm_sweep(config, rpm_range_tuple, d_idx):
    # Nota: Pasamos rpm_range como tupla porque los arrays no son "hasheables"
    rpm_range = np.linspace(rpm_range_tuple[0], rpm_range_tuple[1], rpm_range_tuple[2])
    modelo = SimuladorCentrifuga(config)
    return ejecutar_barrido_rpm(modelo, rpm_range, d_idx)

# ABSTRACCIÓN: Función genérica para dibujar gráficos de respuesta
def plot_response_curves(ax, rpm_range, data, rpm_obj, ejes_config, show_resonances=None):
    """Dibuja curvas de respuesta en frecuencia, línea de operación y resonancias."""
    for eje in ejes_config['todos']:
        ax.plot(rpm_range, data[eje], color=ejes_config['colores'][eje], label=ejes_config['labels'][eje])
    
    ax.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
    
    if show_resonances:
        first_res = True
        for f in show_resonances:
            if f < rpm_range[-1]:
                label = 'Resonancia Teórica' if first_res else ""
                ax.axvline(f, color='red', linestyle='--', alpha=0.5, label=label)
                first_res = False
    
    ax.grid(True, alpha=0.1)
    ax.legend()

# ==========================================#
# 4️⃣ INTERFAZ DE STREAMLIT
# ==========================================#
st.set_page_config(layout="wide")
st.title("Simulador Interactivo de Centrífuga 300F1600 x 700 - Dpto. de Ingeniería de Riera Nadeu")
st.markdown("Modifica los valores en la barra lateral para ver el impacto en las vibraciones.")

# --- BARRA LATERAL ---
st.sidebar.header("Parámetros de Diseño")
m_unbalance = st.sidebar.slider("Masa de Desbalanceo (kg)", 0.1, 8.0, 1.6)
rpm_obj = st.sidebar.number_input("RPM nominales", value=1100)

st.sidebar.subheader("Posición del Sensor (m)")
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
sensor_x = col_s1.number_input("X", value=0.0, step=0.1, format="%.2f")
sensor_y = col_s2.number_input("Y", value=0.8, step=0.1, format="%.2f")
sensor_z = col_s3.number_input("Z", value=0.0, step=0.1, format="%.2f")

st.sidebar.subheader("Configuración del Modelo")
eje_vertical = st.sidebar.selectbox("Eje de Rotación (Vertical)", ('x', 'y', 'z'), index=2)

plano_rotor = ['y', 'z'] if eje_vertical == 'x' else ['x', 'z'] if eje_vertical == 'y' else ['x', 'y']

st.sidebar.subheader("Posición de la Placa (m)")
col_p1, col_p2 = st.sidebar.columns(2)
dist_A = col_p1.number_input(f"Placa {plano_rotor[0].upper()} (dist_A)", value=0.0, step=0.1, format="%.2f")
dist_B = col_p2.number_input(f"Placa {plano_rotor[1].upper()} (dist_B)", value=0.0, step=0.1, format="%.2f")

# --- DICCIONARIO DE CONFIGURACIÓN ---
# OPTIMIZACIÓN: Las matrices de inercia se definen como listas para ser "hasheables"
config_base = {
    "eje_vertical": eje_vertical, 
    "plano_rotor": plano_rotor,
    "excitacion": {
        "distancia_eje": 1.2, "m_unbalance": m_unbalance, "e_unbalance": 0.8
    },
    "placa": {
        "lado_a": 2.4, "lado_b": 2.4, "espesor": 0.1, "radio_agujero": 0.5,
        "dist_A": dist_A, "dist_B": dist_B
    },
    "componentes": {
        "bancada": {"m": 3542, "pos": [0.194, 0, 0.859], "I": [[3235,0,0],[0,3690,0],[0,0,2779]]},
        "motor": {"m": 940, "pos": [1.6, 0, 1.1], "I": [[178,0,0],[0,392,0],[0,0,312]]},
        "cesto": {"m": 1980, "pos": [0.5, 0, 0], "I": [[178,0,0],[0,392,0],[0,0,312]]},
    },
    "tipos_dampers": {
        "ZPVL-235-653_Motor": {"kx": 1.32e6, "ky": 1.32e6, "kz": 1.6e6, "cx": 2.5e4, "cy": 2.5e4, "cz": 5e4},
        "ZPVL-235-453": {"kx": 1.0e6,  "ky": 1.0e6,  "kz": 1.3e6, "cx": 2.5e4, "cy": 2.5e4, "cz": 5e4}
    },
    "dampers": [
        {"tipo": "ZPVL-235-653_Motor", "pos": [1.12, 0.84, 0]},
        {"tipo": "ZPVL-235-653_Motor", "pos": [1.12, -0.84, 0]},
        {"tipo": "ZPVL-235-453", "pos": [-0.93, 0.84, 0]},
        {"tipo": "ZPVL-235-453", "pos": [-0.93, -0.84, 0]}
    ],
    "sensor": {"pos_sensor": [sensor_x, sensor_y, sensor_z]}
}

# --- SELECTOR DE DAMPER ---
st.sidebar.subheader("Selección de Componente")
lista_dampers_config = config_base["dampers"] 
opciones = [f"{i}: {d['tipo']} en {d['pos']}" for i, d in enumerate(lista_dampers_config)]
seleccion = st.sidebar.selectbox("Seleccionar ubicación de damper:", opciones)
d_idx = int(seleccion.split(":")[0])

# --- CONFIGURACIÓN DE LA PROPUESTA ---
st.sidebar.header("Variaciones de la Propuesta")
esp_prop = st.sidebar.slider("Espesor Propuesta [mm]", 40.0, 140.0, 100.0) / 1000
pos_x_motor_prop = st.sidebar.slider("Posición X Motor Propuesta [m]", 1.2, 1.8, 1.6)

config_prop = copy.deepcopy(config_base)
config_prop["placa"]["espesor"] = esp_prop
config_prop["componentes"]["motor"]["pos"][0] = pos_x_motor_prop

# --- EJECUCIÓN DE SIMULACIONES (USANDO CACHÉ) ---
f_res_rpm, _ = cached_natural_frequencies(str(config_base))
f_res_rpm_prop, _ = cached_natural_frequencies(str(config_prop))

rpm_range_tuple = (10, rpm_obj * 1.2, 1000)
results_base = cached_run_rpm_sweep(str(config_base), rpm_range_tuple, d_idx)
results_prop = cached_run_rpm_sweep(str(config_prop), rpm_range_tuple, d_idx)

# Extraer resultados para facilitar acceso
rpm_range = results_base["rpm_range"]
idx_op = np.argmin(np.abs(rpm_range - rpm_obj))

# --- CONFIGURACIÓN DE EJES PARA GRÁFICOS ---
horizontales = config_base["plano_rotor"]
colores = {horizontales[0]: "tab:blue", horizontales[1]: "tab:orange", vertical: "tab:green"}
ejes_lbl = {horizontales[0]: f"Plano 1 ({horizontales[0].upper()})", horizontales[1]: f"Plano 2 ({horizontales[1].upper()})", vertical: f"Vertical ({vertical.upper()})"}
ejes_config = {"vertical": vertical, "todos": [horizontales[0], horizontales[1], vertical], "colores": colores, "labels": ejes_lbl}

# ==========================================#
# 5️⃣ MEMORIA DE CÁLCULO Y LAYOUT
# ==========================================#
st.markdown(f"""
### 📋 Informe de Análisis Dinámico
* **Masa de Desbalanceo:** {m_unbalance:.2f} kg
* **RPM de Operación:** {rpm_obj} RPM
* **Material de la Placa:** Acero (ρ = 7850 kg/m³)
---
""")

col_map1, col_map2 = st.columns([1, 1])

# CORRECCIÓN: Gráfico 2D del mapa siempre coherente (vista superior esquemática)
with col_map1:
    st.write("**Mapa de Ubicación de Dampers (Esquema en plano XY)**")
    fig_map, ax_map = plt.subplots(figsize=(5, 5))

    # Usamos las dimensiones principales para una visualización clara
    lado_a = config_base["placa"]["lado_a"]
    lado_b = config_base["placa"]["lado_b"]
    
    # La posición se toma de la proyección XY del modelo
    modelo_base_temp = SimuladorCentrifuga(config_base) # Se crea un modelo temporal para obtener la pos
    pos_placa_xy = modelo_base_temp.componentes["placa"]["pos"][:2]
    
    anclaje_rect = (pos_placa_xy[0] - lado_a / 2, pos_placa_xy[1] - lado_b / 2)
    
    rect = plt.Rectangle(anclaje_rect, lado_a, lado_b, color='lightgray', alpha=0.3, label='Placa Base')
    ax_map.add_patch(rect)
    
    for i, d in enumerate(modelo_base_temp.dampers):
        ax_map.scatter(d.pos[0], d.pos[1], c='red' if i == d_idx else 'blue', s=150, zorder=3, label='Analizado' if i == d_idx and 'Analizado' not in plt.gca().get_legend_handles_labels()[1] else None)
        ax_map.text(d.pos[0]+0.1, d.pos[1]+0.1, f"D{i}", fontsize=12, fontweight='bold')

    ax_map.scatter(config_base["componentes"]["motor"]["pos"][0], config_base["componentes"]["motor"]["pos"][1], marker='s', s=150, color='gray', alpha=0.5, label='Motor Base', zorder=4)
    ax_map.scatter(pos_x_motor_prop, config_prop["componentes"]["motor"]["pos"][1], marker='s', s=180, color='green', edgecolors='black', label='Motor Propuesta', zorder=5)    

    ax_map.axhline(0, color='black', lw=1); ax_map.axvline(0, color='black', lw=1)
    ax_map.set_xlim(-2, 2); ax_map.set_ylim(-2, 2)
    ax_map.set_xlabel("X [m]"); ax_map.set_ylabel("Y [m]")
    ax_map.grid(True, alpha=0.2); ax_map.legend()
    st.pyplot(fig_map)

with col_map2:
    st.write("**Distribución de Masas (Centro de Gravedad)**")
    _, _, _, cg_final_base = SimuladorCentrifuga(config_base).armar_matrices()
    st.info(f"**CG Global (Base):**\nX: {cg_final_base[0]:.3f} m | Y: {cg_final_base[1]:.3f} m | Z: {cg_final_base[2]:.3f} m")

    _, _, _, cg_final_prop = SimuladorCentrifuga(config_prop).armar_matrices()
    st.info(f"**CG Global (Propuesta):**\nX: {cg_final_prop[0]:.3f} m | Y: {cg_final_prop[1]:.3f} m | Z: {cg_final_prop[2]:.3f} m")

# ==========================================#
# 6️⃣ GRÁFICOS DE RESULTADOS
# ==========================================#
st.subheader("Análisis de Aceleración en el Sensor")
fig1, ax1 = plt.subplots(figsize=(10, 4))
plot_response_curves(ax1, rpm_range, results_base["sensor_aceleracion"], rpm_obj, ejes_config)
ax1.set_xlabel("RPM"); ax1.set_ylabel("Aceleración [g]")
st.pyplot(fig1)

st.subheader("Respuesta en Frecuencia: Velocidad en Sensor")
fig2, ax2 = plt.subplots(figsize=(10, 5))
plot_response_curves(ax2, rpm_range, results_base["sensor_velocidad"], rpm_obj, ejes_config, show_resonances=f_res_rpm)
ax2.set_xlabel('Velocidad de Rotación [RPM]'); ax2.set_ylabel('Velocidad [mm/s]')
st.pyplot(fig2)

st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.subheader(f"Desplazamiento Amplitud en Damper {lista_dampers_config[d_idx]['tipo']}")
fig3, ax3 = plt.subplots(figsize=(10, 5))
plot_response_curves(ax3, rpm_range, results_base["damper_desplazamiento"], rpm_obj, ejes_config)
ax3.set_xlabel('Velocidad de Rotación [RPM]'); ax3.set_ylabel('Desplazamiento [mm]')
st.pyplot(fig3)

st.subheader(f"Fuerzas Dinámicas en Damper {lista_dampers_config[d_idx]['tipo']}")
fig4, ax4 = plt.subplots(figsize=(10, 5))
plot_response_curves(ax4, rpm_range, results_base["damper_fuerza"], rpm_obj, ejes_config)
ax4.set_xlabel('Velocidad de Rotación [RPM]'); ax4.set_ylabel('Fuerza [N]')
st.pyplot(fig4)

# ==========================================#
# 7️⃣ GRÁFICOS COMPARATIVOS
# ==========================================#
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.title("Comparativa de Propuestas")
st.subheader("Comparativa de velocidad en el Sensor")

fig5, ax5 = plt.subplots(figsize=(10, 4))
for eje in ejes_config['todos']:
    ax5.plot(rpm_range, results_base["sensor_velocidad"][eje], color=ejes_config['colores'][eje], linestyle="--", alpha=0.7, label=f"Base {ejes_config['labels'][eje]}")
    ax5.plot(rpm_range, results_prop["sensor_velocidad"][eje], color=ejes_config['colores'][eje], label=f"Propuesta {ejes_config['labels'][eje]}")
ax5.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax5.set_xlabel("RPM"); ax5.set_ylabel("Velocidad [mm/s]"); ax5.legend(); ax5.grid(True, alpha=0.1)
st.pyplot(fig5)

# ==========================================#
# 8️⃣ CONCLUSIONES Y REPORTE
# ==========================================#
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.header("Análisis de Seguridad y Vibraciones")

col_concl1, col_concl2 = st.columns(2)
with col_concl1:
    st.write("### 🚨 Puntos Críticos (Resonancia)")
    st.write(f"**Caso Base (Modo 1):** {f_res_rpm[0]:.0f} RPM | **(Modo 6):** {f_res_rpm[5]:.0f} RPM")
    st.write(f"**Propuesta (Modo 1):** {f_res_rpm_prop[0]:.0f} RPM | **(Modo 6):** {f_res_rpm_prop[5]:.0f} RPM")
    
    dist_min_prop = np.min(np.abs(f_res_rpm_prop - rpm_obj))
    if dist_min_prop < 150:
        st.error(f"⚠️ PELIGRO: Resonancia crítica en la propuesta. Margen insuficiente ({dist_min_prop:.0f} RPM) respecto a {rpm_obj} RPM.")
    else:
        st.success(f"✅ SEGURO: La propuesta mantiene un margen > 150 RPM respecto a la operación (Margen actual: {dist_min_prop:.0f} RPM).")

with col_concl2:
    st.write("### 📊 Cumplimiento de Norma (ISO 10816)")
    v_max_base = max(max(v) for v in results_base["sensor_velocidad"].values())
    v_max_prop = max(max(v) for v in results_prop["sensor_velocidad"].values())
    
    st.write(f"**Velocidad Máx. Base:** {v_max_base:.2f} mm/s")
    st.write(f"**Velocidad Máx Propuesta:** {v_max_prop:.2f} mm/s")
    
    if v_max_prop < 4.5:
        st.success(f"**Clase A/B:** {v_max_prop:.2f} mm/s (Excelente/Satisfactorio)")
    elif v_max_prop < 11.2:
        st.warning(f"**Clase C:** {v_max_prop:.2f} mm/s (Alerta/Insatisfactorio)")
    else:
        st.error(f"**Clase D:** {v_max_prop:.2f} mm/s (Peligro)")

    mejora_vel = ((v_max_base - v_max_prop) / v_max_base) * 100 if v_max_base > 0 else 0
    if mejora_vel > 0:
        st.write(f"📈 Reducción de vibración: **{mejora_vel:.1f}%**")
    else:
        st.write(f"📉 Incremento de vibración: **{abs(mejora_vel):.1f}%**")

st.write("---")
st.subheader("📝 Notas del Analista")
st.text_area("Escribe aquí tus conclusiones adicionales:", "Se observa que el aumento del espesor de la placa desplaza la frecuencia natural hacia arriba, reduciendo la amplitud en el punto de operación.", height=100)

st.divider()
st.subheader("🖨️ Generar Reporte Técnico")
if st.button("Preparar Informe para PDF"):
    st.balloons()
    st.info("### Instrucciones para un PDF Profesional:\n"
            "1. Presiona **Ctrl + P** (Windows) o **Cmd + P** (Mac).\n"
            "2. Selecciona **'Guardar como PDF'**.\n"
            "3. En 'Más ajustes', activa **'Gráficos de fondo'**.\n"
            "4. Cambia el diseño a **'Vertical'**.")
