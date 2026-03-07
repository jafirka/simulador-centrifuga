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
from fpdf import FPDF
from scipy.spatial import ConvexHull



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
        self.tipo = config.get("tipo_de_maquina")   
        self.pos_sensor = np.array(config["sensor"]["pos_sensor"])
        self.excitacion = config['excitacion']

        # --- Componentes ---
        self.componentes = {
            "cesto": config['componentes']['cesto'],
            "bancada": config['componentes']['bancada'],
            "motor": config['componentes']['motor']
        }

        # --- Parámetros de la Placa ---
        if self.tipo == "vertical":
            p = config['placa']
            l_a, l_b, esp = p['lado_a'], p['lado_b'], p['espesor']
            dist_x = p.get('Dist_x', 0.0)
            dist_z = p.get('Dist_z', 0.0)

            r = p['radio_agujero']
            rho = 7850 # kg/m^3 (Acero)
            
            # Mapeo de dimensiones: dx=X, dy=Y(Vertical), dz=Z
            dx, dy, dz = l_a, esp, l_b
            self.dims = {'x': dx, 'y': dy, 'z': dz}
            self.pos_placa = [dist_x, 0.0, dist_z]
            
            # Masa: Masa total - Masa del agujero
            m_total = (dx * dy * dz) * rho
            m_agujero = (math.pi * r**2 * esp) * rho
            m_placa = m_total - m_agujero

            # Cálculo de Inercias Locales (Respecto al CG de la placa)
            # Eje Y: Polar
            Iy = (1/12) * m_total * (l_a**2 + l_b**2) - (1/2) * m_agujero * r**2
            # Eje X: Diametral
            Ix = (1/12) * m_total * (l_b**2 + esp**2) - (1/4) * m_agujero * (r**2 + (esp**2)/3)
            # Eje Z: Diametral
            Iz = (1/12) * m_total * (l_a**2 + esp**2) - (1/4) * m_agujero * (r**2 + (esp**2)/3)


            self.componentes["placa"] = {
                "m": m_placa, 
                "pos": [p.get('Dist_x', 0.0), 0.0, p.get('Dist_z', 0.0)], 
                "I": [[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]]
            }

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

            I_local = np.array(c["I"], dtype=float)

            # Vector desde el CG global al CG del componente
            d = p_c - cg_global
        
            # Teorema de Steiner (Ejes Paralelos) en forma matricial
            # I_global = sum( I_local + m * [ (d·d)diag(1) - (d ⊗ d) ] )
            term_steiner = m_c * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

            # Verificación de simetría real antes de sumar
            matriz_c = I_local + term_steiner
            I_global += (I_local + term_steiner)

		# Verificación final (Solo si la asimetría es grosera)
        if not np.allclose(I_global, I_global.T, atol=1e-6):
            st.error(f"Error crítico: I_global sigue siendo asimétrica.")
            st.matrix(I_global) # st.matrix es genial para ver tensores
			

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

def ejecutar_barrido_rpm(modelo, rpm_range, d_idx,usar_giroscopico=False, i_producto=0.0):

    M, K, C, cg_global = modelo.armar_matrices()
    T_sensor = modelo.obtener_matriz_sensor(cg_global)

    # --- Preparación damper específico ---
    damper_d = modelo.dampers[d_idx]
    T_damper = damper_d.get_matriz_T(cg_global)
    ks = [damper_d.kx, damper_d.ky, damper_d.kz]
    cs = [damper_d.cx, damper_d.cy, damper_d.cz]

    ex = modelo.excitacion

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

        # 1. Definición del Vector de Carga F según orientación
        lx, ly, lz = -cg_global[0], -cg_global[1], -cg_global[2]
        dist = ex['distancia_eje']

        G = np.zeros((6, 6))
        val_g = i_producto * w
        if modelo.tipo == "vertical": # <-- CORREGIDO: Usamos modelo.tipo
            ly_exc = dist + ly # Altura relativa al CG
            F = np.array([
            F0,                     # Fx (Centrífuga en X)
            0,                      # Fy (Nula en el eje de rotación axial)
            -1j * F0,               # Fz (Centrífuga en Z - desfase 90°)
            - (-1j * F0) * ly_exc,  # Mx = Fy*lz - Fz*ly  -> (0 - Fz*ly)
            F0 * lz - (-1j * F0) * lx, # My = Fz*lx - Fx*lz (Momento Torsional en el eje Y)
            F0 * ly_exc             # Mz = Fx*ly - Fy*lx  -> (Fx*ly - 0)
            ])
            if usar_giroscopico:
                # Acoplamiento Rx (3) y Ry (4)
                # El término es Iz * Omega
                G[3, 5] = val_g 
                G[5, 3] = -val_g

        else:
            lz_exc = dist + lz # Distancia axial relativa al CG
            F = np.array([
                F0,                     # Fx (Real)
                -1j * F0,                # Fy (Imaginaria - Giro 90°)
                0,                      # Fz (Nula en desbalanceo radial)
                (1j * F0) * lz_exc,     # Mx = Fy*lz - Fz*ly
                -F0 * lz_exc,           # My = Fz*lx - Fx*lz
                F0 * ly - (1j * F0) * lx  # Mz = Fx*ly - Fy*lx (Momento Torsional)
            ])

            # --- MATRIZ GIROSCÓPICA G ---
            if usar_giroscopico:
                # Acoplamiento Rx (3) y Ry (4)
                # El término es Iz * Omega
                G[3, 4] = val_g
                G[4, 3] = -val_g


        # Resolver el sistema: Z * X = F
        #Z = -w**2 * M + 1j*w * C + K
        Z = -w**2 * M + 1j * w * (C + G) + K
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

def dibujar_modelo_2d_vertical(modelo, titulo="Disposición de Planta (Plano XZ)"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Obtener datos del modelo
    p = modelo.dims
    pos_p = modelo.componentes["placa"]["pos"]

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




def dibujar_modelo_2d_horizontal(modelo, titulo="Disposición Física - Centrífuga Horizontal"):
    # Creamos la figura con dos vistas: Perfil (X-Y) y Planta (X-Z)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Obtención de datos maestros del modelo
    _, _, _, cg_global = modelo.armar_matrices()
    ex = modelo.excitacion
    radio_cesto = ex.get('e_unbalance', 0.625)
    distancia_axial_total = ex.get('distancia_eje', 1.5) # Longitud del rotor
    
    # --- VISTA FRONTAL / PERFIL (PLANO X-Y) ---
    # Representación del cesto sombreado (Círculo)
    cesto_frontal = plt.Circle((0, 0), radio_cesto, color='blue', fill=True, alpha=0.1, label='Cesto (Ø)')
    cesto_borde = plt.Circle((0, 0), radio_cesto, color='blue', fill=False, ls='--', alpha=0.5)
    ax1.add_patch(cesto_frontal)
    ax1.add_patch(cesto_borde)
    
    # Dibujar Dampers en Frontal
    for i, d in enumerate(modelo.dampers):
        ax1.scatter(d.pos[0], d.pos[1], marker='s', s=100, color='black', zorder=5,
                    label="Aisladores (Dampers)" if i==0 else "")
        ax1.text(d.pos[0], d.pos[1] - 0.15, d.nombre, fontsize=8, ha='center', fontweight='bold')

    # Centro de Gravedad Global en Frontal
    ax1.scatter(cg_global[0], cg_global[1], marker='*', s=300, color='gold', 
                edgecolor='black', zorder=10, label=f'CG Total ({cg_global[0]:.2f}, {cg_global[1]:.2f})')
    
    ax1.set_title("Vista Frontal (Sección X-Y)", fontsize=12, pad=10)
    ax1.set_xlabel("Eje X [m]")
    ax1.set_ylabel("Eje Y (Altura) [m]")
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.3)

    # --- VISTA DE PLANTA (PLANO X-Z) ---
    # 2. Área sombreada entre Dampers (Corregida con ConvexHull)
    if len(modelo.dampers) >= 3:
        puntos_xz = np.array([[d.pos[0], d.pos[2]] for d in modelo.dampers])
        try:
            hull = ConvexHull(puntos_xz)
            # Ordenamos los puntos perimetralmente para que el sombreado no se cruce
            puntos_ordenados = puntos_xz[hull.vertices]
            poly_xz = plt.Polygon(puntos_ordenados, closed=True, color='gray', alpha=0.2, label='Área de Apoyo')
            ax2.add_patch(poly_xz)
        except:
            # Fallback simple si los puntos son colineales o hay error
            pass

    # Representación del cesto en planta (Rectángulo)
    rect_cesto = plt.Rectangle((-radio_cesto, 0), radio_cesto*2, distancia_axial_total, 
                               color='blue', alpha=0.1, label='Cuerpo del Cesto')
    ax2.add_patch(rect_cesto)
    
    # Eje de rotación (Línea central)
    ax2.plot([0, 0], [0, distancia_axial_total], color='blue', ls='-.', lw=1.5, alpha=0.6)

    # Dibujar Dampers en Planta
    for d in modelo.dampers:
        ax2.scatter(d.pos[0], d.pos[2], marker='s', s=100, color='black', zorder=5)

    # 3. Representación del Desbalanceo (Masa) en Planta
    if ex['m_unbalance'] > 0:
        pos_z_unb = ex['distancia_eje'] # Posición axial
        radio_unb = ex['e_unbalance']    # Posición radial (en X)
        
        # Punto rojo de la masa
        ax2.scatter(radio_unb, pos_z_unb, color='red', s=150, edgecolor='black', 
                    zorder=15, label=f"Desbalance ({ex['m_unbalance']}kg)")
        # Línea de radio
        ax2.plot([0, radio_unb], [pos_z_unb, pos_z_unb], color='red', ls='--', lw=2)

    # CG Global en Planta
    ax2.scatter(cg_global[0], cg_global[2], marker='*', s=300, color='gold', edgecolor='black', zorder=10)
    
    ax2.set_title("Vista de Planta (Superior X-Z)", fontsize=12, pad=10)
    ax2.set_xlabel("Eje X [m]")
    ax2.set_ylabel("Eje Z (Longitud axial) [m]")
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.3)

    # Configuración de leyenda y títulos
    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=0.98)
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Combinar leyendas únicas
    by_label = dict(zip(labels + labels2, handles + handles2))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
               ncol=3, fontsize='medium', frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

def generar_pdf(config_base, f_res, tabla_fuerzas, fig_planta, fig_vibraciones):
    # Con fpdf2 no necesitas especificar 'Arial', usa 'helvetica' que es estándar
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    
    # Título
    pdf.cell(0, 10, "Informe Tecnico de Vibraciones - Riera Nadeu", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    
    # --- GRÁFICO 1: DISPOSICIÓN ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Disposicion Fisica del Sistema", new_x="LMARGIN", new_y="NEXT")
    
    img_buf = io.BytesIO()
    fig_planta.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    pdf.image(img_buf, x=10, w=100) # fpdf2 maneja el buffer automáticamente
    pdf.ln(5)

    # --- GRÁFICO 2: VIBRACIONES ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "2. Analisis de Vibraciones (Fuerza vs Tiempo)", new_x="LMARGIN", new_y="NEXT")
    
    img_buf_2 = io.BytesIO()
    fig_vibraciones.savefig(img_buf_2, format='png', bbox_inches='tight')
    img_buf_2.seek(0)
    pdf.image(img_buf_2, x=10, w=180)

    # --- TABLA DE FUERZAS ---
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "3. Reacciones en Apoyos", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8)
    
    # Encabezados
    pdf.cell(45, 7, "Damper", border=1)
    pdf.cell(45, 7, "Carga Est. [N]", border=1)
    pdf.cell(45, 7, "Carga Tot. Max [N]", border=1)
    pdf.ln()
    
    for _, row in tabla_fuerzas.iterrows():
        pdf.cell(45, 7, str(row["Damper"]), border=1)
        pdf.cell(45, 7, str(row["Carga Estática [N]"]), border=1)
        pdf.cell(45, 7, str(row["Carga TOTAL MÁX [N]"]), border=1)
        pdf.ln()

    # SOLUCIÓN AL ERROR: Simplemente output() sin encode
    return pdf.output()