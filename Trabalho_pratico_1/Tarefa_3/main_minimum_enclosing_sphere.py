#!/usr/bin/env python3
# shebang line for linux / mac
# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import copy
import time
import threading
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from functools import partial
import pyvista as pv
import pyvistaqt as pvqt
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
import sys
import miniball

# Dicionário com parâmetros da camâra
view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory": [
        {
            "boundingbox_max": [2.1948, 0.9802, 4.4420],    # limite máx de visualização
            "boundingbox_min": [-2.5293, -1.1011, 1.4601],  # limite min de visualização
            "field_of_view": 60.0,                          # FOV
            "front": [-0.4499, 0.4272, -0.7842],
            "lookat": [-1.5100, 0.2870, 1.6367],
            "up": [0.0037, -0.8772, -0.4800],
            "zoom": 0.3459,
        }
    ],
    "version_major": 1,
    "version_minor": 0
}

# Carregar imagens rgb e depth e criar imagem rgbd
filename_rgb1 = "./tum_dataset/rgb/1.png"
filename_depth1 = "./tum_dataset/depth/1.png"
rgb1 = o3d.io.read_image(filename_rgb1)
depth1 = o3d.io.read_image(filename_depth1)
rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)

filename_rgb2 = "./tum_dataset/rgb/2.png"
filename_depth2 = "./tum_dataset/depth/2.png"
rgb2 = o3d.io.read_image(filename_rgb2)
depth2 = o3d.io.read_image(filename_depth2)
rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)

# parâmetros intrínsecos da câmara para criar nuvem de pontos (formato TUM)
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

# criar nuvem de pontos através de imagens rgbd
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intrinsics)
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intrinsics)

# Downsample das nuvens para reduzir esforço computacional
voxel_size = 0.025
dpcd1 = copy.deepcopy(pcd1).voxel_down_sample(voxel_size)
dpcd2 = copy.deepcopy(pcd2).voxel_down_sample(voxel_size)

# Lista de cores possíveis para a esfera
SPHERE_COLORS = [
    [1, 0, 0],      # vermelho
    [0, 0, 1],      # azul
    [1, 1, 0],      # amarelo
    [0, 1, 0],      # verde
]

# global index para ciclo de cores
current_color_index = 0

# função para aplicar cor à esfera (geom)
def apply_color_to_geom(geom, color_index): 
    color = SPHERE_COLORS[color_index]
    geom.paint_uniform_color(color)


# tenta carregar a transformação final para as nuvens de pontos de um txt
try:
    with open(r"output_opti.txt", "r") as file:
        content = file.read()
    values = [float(x) for x in content.strip("[] \n").split(",")]
    params_from_file = values
    # Converte assuming params = [rx,ry,rz,tx,ty,tz]

    def translation_rotation_to_transformation(rot_trans):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rot_trans[:3]).as_matrix()
        T[:3, 3] = rot_trans[3:]
        return T
    
    trans_opti = translation_rotation_to_transformation(params_from_file) # conversão para matriz
    pcd_source_opti = copy.deepcopy(pcd2)
    pcd_source_opti.transform(trans_opti)
    print("Transformação final obtida pelo ficheiro")

# caso o ficheiro não exista usa point cloud 2 original
except Exception as e:   
    print("Ficheiro com transformação final não encontrado, utilizar nuvem na posição inicial.")
    pcd_source_opti = copy.deepcopy(pcd2)

#função para criar a esfera englobante
def create_sphere_pointcloud_from_array(center_with_radius, n_points=2000, include_interior=False, color=[1,0,0], rng=None):
    rng = np.random.default_rng(rng) #manter os pontos da esfera no mesmo lugar
    center = np.asarray(center_with_radius[:3])
    radius = float(center_with_radius[3])
    dirs = rng.normal(size=(n_points,3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    if include_interior:
        r = (rng.random(n_points) ** (1/3)) * radius
    else:
        r = np.full(n_points, radius)
    points = center + dirs * r[:,None]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd

def compute_initial_sphere_params(pcd):
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    radius = dists.max()
    return np.append(center, radius)

def error_target_source(pcd, sphere_params, lambda_r=20): # lamda_r 20 para r grandes, proximo 1 de para r pequeno
    points = np.asarray(pcd.points)
    center = sphere_params[:3]
    radius = sphere_params[3]
    dists = np.linalg.norm(points - center, axis=1)
    excess = np.maximum(0, dists - radius)
    error_points = np.sum(excess)
    error_radius = lambda_r * radius
    print('points ',error_points, 'radius', radius)
    error = float(error_points + error_radius)
    return error

# ----------------- Preparar dados para otimização -----------------
pcd_total = copy.deepcopy(pcd1) + copy.deepcopy(pcd_source_opti)
initial_params = compute_initial_sphere_params(pcd_total)
#initial_params = [0, 0, 0, 1]
#initial_params = [-0.22830394, -0.27319147, 3.69128634, 2.42003145]
print("Parâmetros inicias esfera (x,y,z,r):", initial_params)

pcd_sphere_initial = create_sphere_pointcloud_from_array(initial_params)

# Dicionário usado para comunicar com o thread do visualizador
shared = {
    'latest_sphere_points': np.asarray(pcd_sphere_initial.points).copy(),
    'stop': False,
    'result': None,
    'snapshots': [],       # snapshots para as diferentes janelas do menu final
    'final_params': None
}

# -------------- Função objetivo --------------
def objectiveFunction(params, shared_mem):
    # params = [xc, yc, zc, r]
    center_with_radius = params
    # gerar pontos da esfera (superfície) a partir de params    
    pcd = create_sphere_pointcloud_from_array(center_with_radius, rng = 1)
    points = np.asarray(pcd.points)    # rng = np.random.default_rng(0)
    # dirs = rng.normal(size=(2000,3))
    # dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # r = np.full(2000, float(center_with_radius[3]))
    # points = center_with_radius[:3] + dirs * r[:,None]
    # atualização dos pontos da esfera para o display
    shared_mem['latest_sphere_points'] = points

    # guardar os primeiros 3 snapshots (para menu final)
    if len(shared_mem['snapshots']) < 3:
        shared_mem['snapshots'].append(points.copy())

    # calcular o erro 
    error = error_target_source(pcd_total, center_with_radius)
    #print('Erro = ' + str(error))
    return error

# -------------- Thread que executa a otimização ----------------
def run_optimization(initial_params, shared_local):
    print("Iniciando otimização...")
    res = least_squares(partial(objectiveFunction, shared_mem=shared_local), initial_params, verbose=0, max_nfev=200)
    shared_local['result'] = res
    shared_local['final_params'] = res.x

# -------------- Visualização em tempo real  --------------
vis = o3d.visualization.VisualizerWithKeyCallback()  # com keycallback para possibilitar o uso de teclas com funções
vis.create_window("Sphere optimization (real-time)", width=1280, height=720)

# Adicionar geometria estática (pcd_total) e a esfera que atualiza ao longo da otimização
total_geom = copy.deepcopy(pcd_total)
sphere_geom = copy.deepcopy(pcd_sphere_initial)

vis.add_geometry(total_geom)
vis.add_geometry(sphere_geom)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(axes)

# aplicar camera do 'view'
traj = view["trajectory"][0]
ctr = vis.get_view_control()
ctr.set_front(traj["front"])
ctr.set_lookat(traj["lookat"])
ctr.set_up(traj["up"])
ctr.set_zoom(traj["zoom"])

# Key callback para alternar cor (tecla 'C') na janela em tempo real
def change_color_realtime(vis_local):
    global current_color_index, sphere_geom
    current_color_index = (current_color_index + 1) % len(SPHERE_COLORS)
    apply_color_to_geom(sphere_geom, current_color_index)
    vis_local.update_geometry(sphere_geom)
    return False

vis.register_key_callback(ord("C"), change_color_realtime)  # ligar a tecla c à função de mudança de cor

# Iniciar thread de otimização (background)
opt_thread = threading.Thread(target=run_optimization, args=(initial_params, shared), daemon=True)
opt_thread.start()

# Loop principal (render) — corre no thread principal e atualiza pontos da esfera in-place
try:
    while opt_thread.is_alive():
        if shared['stop']:
            break
        latest = shared.get('latest_sphere_points', None)
        if latest is not None:
            np.asarray(sphere_geom.points)[:] = latest
            vis.update_geometry(sphere_geom)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)

    # garantir a última atualização após fim da otimização
    final_pts = shared.get('latest_sphere_points', None)
    if final_pts is not None:
        np.asarray(sphere_geom.points)[:] = final_pts
        vis.update_geometry(sphere_geom)
        vis.poll_events()
        vis.update_renderer()

    # guardar parâmetros de visualização para garantir uma troca de janelas
    saved_camera = ctr.convert_to_pinhole_camera_parameters()

    print("Pressiona ESC ou fecha a janela para sair.")
    
    vis.run()
finally:
    vis.destroy_window()

# ---------------- After optimization get final sphere and different stages of optimization ----------------

if shared.get('final_params') is not None:
    final_params = shared['final_params']
    pcd_sphere_final = create_sphere_pointcloud_from_array(final_params, color=[0.8,0.2,0.2])
    print("Otimização terminada. Resultados (x,y,z,r):", final_params)
    with open("output_opti_sphere.txt", "w") as file:
        file.write(str(final_params))
else:
    pcd_sphere_final = copy.deepcopy(sphere_geom)  # fallback

# snapshots (se houver) -> converter para PointClouds
pcd_sphere_snapshots = []
for snap_pts in shared.get('snapshots', []):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(snap_pts)
    p.paint_uniform_color([1,0.6,0])
    pcd_sphere_snapshots.append(p)

# ----------------- Menu final (estilo ICPViewer) -----------------
class FinalMenu(QMainWindow):
    def __init__(self, camera_params):
        super().__init__()
        self.setWindowTitle("Resultados e Visualizações (Menu)")
        self.setGeometry(200, 100, 420, 360)
        self.camera_params = camera_params
        self.color_mode = "real"  

        layout = QVBoxLayout()   # layout vertical para botões
        layout.addWidget(QLabel("Ver passos da otimização"))

        # Botões para visualização dos passos da otimização
        btn1 = QPushButton("Point clouds iniciais (desalinhadas)")
        btn1.clicked.connect(lambda: self.show_scene(1))
        layout.addWidget(btn1)

        btn2 = QPushButton("Point clouds alinhadas")
        btn2.clicked.connect(lambda: self.show_scene(2))
        layout.addWidget(btn2)

        btn3 = QPushButton("Esfera inicial")
        btn3.clicked.connect(lambda: self.show_scene(3))
        layout.addWidget(btn3)

        btn4 = QPushButton("Esfera final")
        btn4.clicked.connect(lambda: self.show_scene(4))
        layout.addWidget(btn4)

        layout.addWidget(QLabel("Nota: tecla 'C' alterna ciclo de cores na janela ativa."))

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def show_scene(self, idx):
        vis2 = o3d.visualization.VisualizerWithKeyCallback()
        title = ""
        sphere_copy = None

        if idx == 1:  # 1ª janela mostra as nuvens de pontos originais
            title = "Point clouds iniciais"
            src = copy.deepcopy(pcd2)   
            tgt = copy.deepcopy(pcd1)
            vis2.create_window(window_name=title, width=1280, height=720)
            vis2.add_geometry(tgt)
            vis2.add_geometry(src)

        elif idx == 2: # 2ª janela mostra as nuvens de pontos alinhadas
            title = "Point clouds alinhadas"
            src = copy.deepcopy(pcd_source_opti)
            tgt = copy.deepcopy(pcd1)
            vis2.create_window(window_name=title, width=1280, height=720)
            vis2.add_geometry(tgt)
            vis2.add_geometry(src)

        elif idx == 3: # 3ª janela mostra as nuvens de pontos englobadas pela primeira iteração da esfera
            title = "Esfera inicial"
            vis2.create_window(window_name=title, width=1280, height=720)
            vis2.add_geometry(copy.deepcopy(total_geom))
            sphere_copy = copy.deepcopy(create_sphere_pointcloud_from_array(initial_params))
            vis2.add_geometry(sphere_copy)

        elif idx == 4: # 4ª janela mostra a iteração final das nuvens de pontos englobadas pela esfera de raio menor
            title = "Esfera final"
            vis2.create_window(window_name=title, width=1280, height=720)
            vis2.add_geometry(copy.deepcopy(total_geom))
            sphere_copy = copy.deepcopy(pcd_sphere_final)
            vis2.add_geometry(sphere_copy)

        # aplicar último ponto de visualização
        ctr2 = vis2.get_view_control()
        if self.camera_params is not None:
            try:
                ctr2.convert_from_pinhole_camera_parameters(self.camera_params)
            except:
                pass


            # bind da tecla 'C' para alternar cor ciclicamente nesta janela
            def change_color_vis2(vis_local):
                global current_color_index
                current_color_index = (current_color_index + 1) % len(SPHERE_COLORS)
                apply_color_to_geom(sphere_copy, current_color_index)
                vis_local.update_geometry(sphere_copy)
                return False

            vis2.register_key_callback(ord("C"), change_color_vis2)

        vis2.run()
        # atualiza camera guardada a partir desta janela (permite preservar orientação na próxima)
        try:
            self.camera_params = ctr2.convert_to_pinhole_camera_parameters()
        except:
            pass
        vis2.destroy_window()

# Iniciar aplicação Qt com menu final
app = QApplication(sys.argv)
menu = FinalMenu(camera_params=saved_camera if 'saved_camera' in locals() else None)
menu.show()
sys.exit(app.exec_())





