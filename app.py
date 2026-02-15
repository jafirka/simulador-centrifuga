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
        
        p = config['placa']
        l_a, l_b, esp = p['lado_a'], p['lado_b'], p['espesor']
        dist_A, dist_B = p.get('dist_A', 0), p.get('dist_B', 0)
        r = p['radio_agujero']
        rho = 7850

        if self.eje_vertical == 'z':
            dx, dy, dz = l_a, l_b, esp
        elif self.eje_vertical == 'y':
            dx, dy, dz = l_a, esp, l_b
        else:
            dx, dy, dz = esp, l_a, l_b
        self.dims = {'x': dx, 'y': dy, 'z': dz}

        m_total = (dx * dy * dz) * rho
        m_agujero = (math.pi * r**2 * esp) * rho
        self.m_placa = m_total - m_agujero

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

        if self.eje_vertical == 'z':
            pos_placa = [dist_A, dist_B, 0]
        elif self.eje_vertical == 'y':
            pos_placa = [dist_A, 0, dist_B]
        else:
            pos_placa = [0, dist_A, dist_B]

        self.componentes = {
            "placa": {"m": self.m_placa, "pos": pos_placa, "I": self.I_placa},
            "cesto": config['componentes']['cesto'],
            "bancada": config['componentes']['bancada'],
            "motor": config['componentes']['motor']
        }
        
        for comp_dict in self.componentes.values():
            if "I" in comp_dict and isinstance(comp_dict["I"], list):
                comp_dict["I"] = np.array(comp_dict["I"])

        self.excitacion = config['excitacion']
        self.dampers = []
        for d_conf in config['dampers']:
            tipo_nombre = d_conf['tipo']
            tipo_vals = config['tipos_dampers'][tipo_nombre]
            self.dampers.append(Damper(tipo_nombre, d_conf['pos'], **tipo_vals))

    def obtener_matriz_sensor(self, cg_global):
        r_p = self.pos_sensor - cg_global
        return np.array([[1,0,0,0,r_p[2],-r_p[1]],[0,1,0,-r_p[2],0,r_p[0]],[0,0,1,r_p[1],-r_p[0],0]])

    def armar_matrices(self):
        m_total = sum(c["m"] for c in self.componentes.values())
        cg_global = sum(c["m"] * np.array(c["pos"]) for c in self.componentes.values()) / m_total

        M, I_global = np.zeros((6, 6)), np.zeros((3, 3))
        for c in self.componentes.values():
            I_local = np.diag(c["I"]) if len(np.array(c["I"]).shape) == 1 else c["I"]
            d = np.array(c["pos"]) - cg_global
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
        M, K, _, _ = self.armar_matrices()
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
    ks, cs = [damper_d.kx,damper_d.ky,damper_d.kz], [damper_d.cx,damper_d.cy,damper_d.cz]

    ex = modelo.excitacion
    dist = ex['distancia_eje']
    
    results = {
        "acel_cg": {"x":[],"y":[],"z":[]}, "vel_cg": {"x":[],"y":[],"z":[]},
        "D_desp": {"x":[],"y":[],"z":[]}, "D_fuerza": {"x":[],"y":[],"z":[]},
        "S_desp": {"x":[],"y":[],"z":[]}, "S_vel": {"x":[],"y":[],"z":[]}, "S_acel": {"x":[],"y":[],"z":[]}
    }

    for rpm in rpm_range:
        w = rpm * 2 * np.pi / 60
        F0 = ex['m_unbalance'] * ex['e_unbalance'] * w**2

        arm = dist - cg_global[{"x": 0, "y": 1, "z": 2}[modelo.eje_vertical]]
        if modelo.eje_vertical == 'x': F = np.array([0,F0,F0*1j,0,-(F0*1j)*arm,F0*arm])
        elif modelo.eje_vertical == 'y': F = np.array([F0,0,F0*1j,(F0*1j)*arm,0,-F0*arm])
        else: F = np.array([F0,F0*1j,0,(F0*1j)*arm,-F0*arm,0])

        Z = -w**2 * M + 1j*w * C + K
        X = linalg.solve(Z, F)
        
        X_damper, U_sensor = T_damper @ X, T_sensor @ X

        for i, eje in enumerate(["x", "y", "z"]):
            results["acel_cg"][eje].append((w**2) * np.abs(X[i]) / 9.81)
            results["vel_cg"][eje].append(w * np.abs(X[i]) * 1000)
            results["D_desp"][eje].append(np.abs(X_damper[i]) * 1000)
            results["D_fuerza"][eje].append(np.abs((ks[i] + 1j*w*cs[i]) * X_damper[i]))
            results["S_desp"][eje].append(np.abs(U_sensor[i]) * 1000)
            results["S_vel"][eje].append(w * np.abs(U_sensor[i]) * 1000)
            results["S_acel"][eje].append((w**2) * np.abs(U_sensor[i]) / 9.81)
    
    results["rpm_range"] = rpm_range
    return results

# ==========================================#
# 3️⃣ FUNCIONES AUXILIARES Y DE CACHÉ
# ==========================================#

@st.cache_data
def cached_natural_frequencies(config):
    modelo = SimuladorCentrifuga(config)
    return modelo.calcular_frecuencias_naturales()

@st.cache_data
def cached_run_rpm_sweep(config, rpm_range_tuple, d_idx):
    rpm_range = np.linspace(rpm_range_tuple[0], rpm_range_tuple[1], rpm_range_tuple[2])
    modelo = SimuladorCentrifuga(config)
    return ejecutar_barrido_rpm(modelo, rpm_range, d_idx)

def plot_response_curves(ax, rpm_range, data, rpm_obj, ejes_config, show_resonances=None):
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
    ax.grid(True, alpha=0.1); ax.legend()

# ==========================================#
# 4️⃣ INTERFAZ DE STREAMLIT
# ==========================================#
st.set_page_config(layout="wide")
st.title("Simulador Interactivo de Centrífuga 300F1600 x 700")

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
dist_A = col_p1.number_input(f"Placa {plano_rotor[0].upper()}", value=0.0, step=0.1, format="%.2f")
dist_B = col_p2.number_input(f"Placa {plano_rotor[1].upper()}", value=0.0, step=0.1, format="%.2f")

# --- DICCIONARIO DE CONFIGURACIÓN ---
config_base = {
    "eje_vertical": eje_vertical, "plano_rotor": plano_rotor,
    "excitacion": {"distancia_eje":1.2, "m_unbalance":m_unbalance, "e_unbalance":0.8},
    "placa": {"lado_a":2.4, "lado_b":2.4, "espesor":0.1, "radio_agujero":0.5, "dist_A":dist_A, "dist_B":dist_B},
    "componentes": {
        "bancada": {"m":3542, "pos":[0.194,0,0.859], "I":[[3235,0,0],[0,3690,0],[0,0,2779]]},
        "motor": {"m":940, "pos":[1.6,0,1.1], "I":[[178,0,0],[0,392,0],[0,0,312]]},
        "cesto": {"m":1980, "pos":[0.5,0,0], "I":[[178,0,0],[0,392,0],[0,0,312]]},
    },
    "tipos_dampers": {
        "ZPVL-235-653_Motor": {"kx":1.32e6, "ky":1.32e6, "kz":1.6e6, "cx":2.5e4, "cy":2.5e4, "cz":5e4},
        "ZPVL-235-453": {"kx":1.0e6, "ky":1.0e6, "kz":1.3e6, "cx":2.5e4, "cy":2.5e4, "cz":5e4}
    },
    "dampers": [
        {"tipo":"ZPVL-235-653_Motor", "pos":[1.12,0.84,0]}, {"tipo":"ZPVL-235-653_Motor", "pos":[1.12,-0.84,0]},
        {"tipo":"ZPVL-235-453", "pos":[-0.93,0.84,0]}, {"tipo":"ZPVL-235-453", "pos":[-0.93,-0.84,0]}
    ],
    "sensor": {"pos_sensor": [sensor_x, sensor_y, sensor_z]}
}

# --- SELECTOR Y PROPUESTA ---
st.sidebar.subheader("Análisis de Componentes")
opciones = [f"{i}: {d['tipo']}" for i,d in enumerate(config_base["dampers"])]
d_idx = int(st.sidebar.selectbox("Seleccionar damper:", opciones).split(":")[0])

st.sidebar.header("Variaciones de la Propuesta")
esp_prop = st.sidebar.slider("Espesor Propuesta [mm]", 40.0, 140.0, 100.0) / 1000
pos_x_motor_prop = st.sidebar.slider("Posición X Motor [m]", 1.2, 1.8, 1.6)

config_prop = copy.deepcopy(config_base)
config_prop["placa"]["espesor"] = esp_prop
config_prop["componentes"]["motor"]["pos"][0] = pos_x_motor_prop

# --- EJECUCIÓN DE SIMULACIONES (USANDO CACHÉ) ---
# CORRECCIÓN: Se eliminó la conversión `str()` que causaba el TypeError
f_res_rpm, _ = cached_natural_frequencies(config_base)
f_res_rpm_prop, _ = cached_natural_frequencies(config_prop)

rpm_range_tuple = (10, rpm_obj * 1.2, 1000)
results_base = cached_run_rpm_sweep(config_base, rpm_range_tuple, d_idx)
results_prop = cached_run_rpm_sweep(config_prop, rpm_range_tuple, d_idx)

rpm_range = results_base["rpm_range"]
idx_op = np.argmin(np.abs(rpm_range - rpm_obj))

# --- CONFIG DE EJES PARA GRÁFICOS ---
horizontales = config_base["plano_rotor"]
colores = {horizontales[0]:"tab:blue", horizontales[1]:"tab:orange", eje_vertical:"tab:green"}
ejes_lbl = {h:f"Plano {i+1} ({h.upper()})" for i,h in enumerate(horizontales)}
ejes_lbl[eje_vertical] = f"Vertical ({eje_vertical.upper()})"
ejes_config = {"todos":[horizontales[0], horizontales[1], eje_vertical], "colores":colores, "labels":ejes_lbl}

# ==========================================#
# 5️⃣ MEMORIA DE CÁLCULO Y LAYOUT
# ==========================================#
col_map1, col_map2 = st.columns([1, 1])

with col_map1:
    st.write("**Mapa de Ubicación (Esquema en plano XY)**")
    fig_map, ax_map = plt.subplots(figsize=(5, 5))
    
    modelo_base_temp = SimuladorCentrifuga(config_base)
    pos_placa_xy = modelo_base_temp.componentes["placa"]["pos"][:2]
    
    rect = plt.Rectangle((pos_placa_xy[0] - config_base["placa"]["lado_a"] / 2, pos_placa_xy[1] - config_base["placa"]["lado_b"] / 2), 
                         config_base["placa"]["lado_a"], config_base["placa"]["lado_b"], 
                         color='lightgray', alpha=0.3, label='Placa Base')
    ax_map.add_patch(rect)
    
    for i, d in enumerate(modelo_base_temp.dampers):
        ax_map.scatter(d.pos[0], d.pos[1], c='red' if i == d_idx else 'blue', s=150, zorder=3, label='Analizado' if i == d_idx else None)
        ax_map.text(d.pos[0]+0.1, d.pos[1]+0.1, f"D{i}", fontsize=10)

    ax_map.scatter(config_base["componentes"]["motor"]["pos"][0], config_base["componentes"]["motor"]["pos"][1], marker='s', s=120, color='gray', label='Motor Base', zorder=4)
    ax_map.scatter(pos_x_motor_prop, config_prop["componentes"]["motor"]["pos"][1], marker='s', s=150, color='green', ec='k', label='Motor Propuesta', zorder=5)    

    ax_map.set_xlim(-2, 2); ax_map.set_ylim(-2, 2)
    ax_map.set_xlabel("X [m]"); ax_map.set_ylabel("Y [m]")
    ax_map.grid(True, alpha=0.2); ax_map.legend()
    st.pyplot(fig_map)

with col_map2:
    st.write("**Centro de Gravedad Global**")
    _, _, _, cg_base = modelo_base_temp.armar_matrices()
    st.info(f"**Base:** X: {cg_base[0]:.3f} | Y: {cg_base[1]:.3f} | Z: {cg_base[2]:.3f} m")
    _, _, _, cg_prop = SimuladorCentrifuga(config_prop).armar_matrices()
    st.info(f"**Prop.:** X: {cg_prop[0]:.3f} | Y: {cg_prop[1]:.3f} | Z: {cg_prop[2]:.3f} m")
    st.markdown(f"**RPM Operación:** {rpm_obj} RPM\n\n**Masa Desbalanceo:** {m_unbalance:.2f} kg")

# ==========================================#
# 6️⃣ GRÁFICOS DE RESULTADOS
# ==========================================#
st.divider()
st.subheader("Resultados del Modelo Base")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_response_curves(ax1, rpm_range, results_base["S_acel"], rpm_obj, ejes_config)
ax1.set_title("Aceleración en Sensor"); ax1.set_ylabel("Aceleración [g]")

plot_response_curves(ax2, rpm_range, results_base["S_vel"], rpm_obj, ejes_config, show_resonances=f_res_rpm)
ax2.set_title("Velocidad en Sensor"); ax2.set_ylabel("Velocidad [mm/s]")

st.pyplot(fig)

# ==========================================#
# 7️⃣ GRÁFICOS COMPARATIVOS
# ==========================================#
st.divider()
st.subheader("Comparativa de Propuestas")
fig_comp, (ax_comp1, ax_comp2) = plt.subplots(1, 2, figsize=(12, 4))

for eje in ejes_config['todos']:
    ax_comp1.plot(rpm_range, results_base["S_vel"][eje], color=ejes_config['colores'][eje], ls="--", alpha=0.7, label=f"Base {ejes_lbl[eje]}")
    ax_comp1.plot(rpm_range, results_prop["S_vel"][eje], color=ejes_config['colores'][eje], label=f"Prop. {ejes_lbl[eje]}")
ax_comp1.axvline(rpm_obj, color='k', ls=':', label=f'RPM op.')
ax_comp1.set_title("Velocidad en Sensor"); ax_comp1.set_xlabel("RPM"); ax_comp1.legend()

for eje in ejes_config['todos']:
    ax_comp2.plot(rpm_range, results_base["D_fuerza"][eje], color=ejes_config['colores'][eje], ls="--", alpha=0.7)
    ax_comp2.plot(rpm_range, results_prop["D_fuerza"][eje], color=ejes_config['colores'][eje])
ax_comp2.axvline(rpm_obj, color='k', ls=':')
ax_comp2.set_title(f"Fuerza en Damper {d_idx}"); ax_comp2.set_xlabel("RPM"); ax_comp2.set_ylabel("Fuerza [N]")

st.pyplot(fig_comp)
