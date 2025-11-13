
import open3d as o3d
import numpy as np
import copy
import time
import threading
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from functools import partial

# ---------- CONFIGURAÇÃO DE CÂMARA ----------
view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory": [
        {
            "boundingbox_max": [2.1948, 0.9802, 4.4420],
            "boundingbox_min": [-2.5293, -1.1011, 1.4601],
            "field_of_view": 60.0,
            "front": [-0.4499, 0.4272, -0.7842],
            "lookat": [-1.5100, 0.2870, 1.6367],
            "up": [0.0037, -0.8772, -0.4800],
            "zoom": 0.3459,
        }
    ],
    "version_major": 1,
    "version_minor": 0
}

# ---------- ABRIR E CONVERTER RGB-D ----------
filename_rgb1 = "SAVI_Mateus/Aula08/tum_dataset/rgb/1.png"
filename_depth1 = "SAVI_Mateus/Aula08/tum_dataset/depth/1.png"
rgb1 = o3d.io.read_image(filename_rgb1)
depth1 = o3d.io.read_image(filename_depth1)
rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)

filename_rgb2 = "SAVI_Mateus/Aula08/tum_dataset/rgb/2.png"
filename_depth2 = "SAVI_Mateus/Aula08/tum_dataset/depth/2.png"
rgb2 = o3d.io.read_image(filename_rgb2)
depth2 = o3d.io.read_image(filename_depth2)
rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)

# ---------- CRIAR PCD ----------
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd1, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd2, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# ---------- DOWNSAMPLING ----------
voxel_size = 0.025
dpcd1 = copy.deepcopy(pcd1).voxel_down_sample(voxel_size)
dpcd2 = copy.deepcopy(pcd2).voxel_down_sample(voxel_size)

# ---------- MATRIZ DE TRANSFORMAÇÃO INICIAL ----------
trans_init = np.asarray([
    [0.9836, -0.0810, 0.1608, -0.9015],
    [0.0799, 0.9967, 0.0131, -0.0011],
    [-0.1613, -0.00006, 0.9868, 0.1361],
    [0.0, 0.0, 0.0, 1.0]
])

# ---------- FUNÇÕES AUXILIARES ----------
def translation_rotation_to_transformation(rot_trans):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rot_trans[:3]).as_matrix()
    T[:3, 3] = rot_trans[3:]
    return T

def transformation_to_translation_rotation(trans):
    R_mat = trans[:3, :3]
    t = trans[:3, 3]
    rotvec = Rotation.from_matrix(R_mat).as_rotvec()
    return np.concatenate([rotvec, t])

def error_target_source_points(src_pts, tgt_pts, keep_ratio=0.90):
    kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts)))
    residuals = np.empty(len(src_pts))
    for i, p in enumerate(src_pts):
        [_, idx, _] = kdtree.search_knn_vector_3d(p, 1)
        diff = p - tgt_pts[idx[0]]
        residuals[i] = np.sum(diff**2)
    n_keep = int(len(residuals) * keep_ratio)
    threshold = np.partition(residuals, n_keep-1)[n_keep-1]
    return residuals[residuals <= threshold]

# ---------- PONTOS ORIGINAIS ----------
target_pts_orig = np.asarray(dpcd1.points)
source_pts_orig = np.asarray(dpcd2.points)

# ---------- SHARED MEMORY ----------
shared = {
    'latest_points': source_pts_orig.copy(),
    'stop': False,
    'result': None,
    'snapshots': []
}

# ---------- FUNÇÃO OBJETIVO ICP ----------
def objective_opt(params, shared_local):
    if shared_local.get("stop"):
        return np.zeros(1)

    T = translation_rotation_to_transformation(params)
    R_mat = T[:3, :3]
    t = T[:3, 3]
    transformed = (R_mat @ source_pts_orig.T).T + t
    shared_local['latest_points'] = transformed

    # Guardar snapshot a cada iteração importante
    if len(shared_local['snapshots']) < 3:
        shared_local['snapshots'].append(transformed.copy())

    residuals = error_target_source_points(transformed, target_pts_orig, keep_ratio=0.90)
    return residuals

# ---------- THREAD DE OTIMIZAÇÃO ----------
def run_optimization(initial_params, shared_local):
    print("Iniciando otimização...")
    res = least_squares(partial(objective_opt, shared_local=shared_local),
                        initial_params, verbose=2, max_nfev=200)
    shared_local['result'] = res
    print("Otimização terminada.")
    return

# ---------- VISUALIZAÇÃO EM TEMPO REAL ----------
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("ICP em tempo real", width=1280, height=720)

target_geom = copy.deepcopy(dpcd1)
source_geom = copy.deepcopy(dpcd2)
target_geom.paint_uniform_color([0, 0.651, 0.929])
source_geom.paint_uniform_color([1, 0.706, 0])
vis.add_geometry(target_geom)
vis.add_geometry(source_geom)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(axes)

traj = view["trajectory"][0]
ctr = vis.get_view_control()
ctr.set_front(traj["front"])
ctr.set_lookat(traj["lookat"])
ctr.set_up(traj["up"])
ctr.set_zoom(traj["zoom"])

# ---------- GESTÃO DE CORES ----------
color_modes = [
    ([1, 0, 0], [0, 0, 1]),     # vermelho / azul
    ([0, 1, 0], [1, 0, 1]),     # verde / roxo
    ([0, 0.651, 0.929], [1, 0.706, 0]),  # azul original / laranja
]
current_color_mode = 2

def apply_colors(mode_index):
    c1, c2 = color_modes[mode_index]
    target_geom.paint_uniform_color(c1)
    source_geom.paint_uniform_color(c2)
    vis.update_geometry(target_geom)
    vis.update_geometry(source_geom)

def toggle_color(vis):
    global current_color_mode
    current_color_mode = (current_color_mode + 1) % len(color_modes)
    apply_colors(current_color_mode)
    print(f"Modo de cor alterado para {current_color_mode + 1}")
    return False

def use_real_colors(vis):
    target_geom.colors = dpcd1.colors
    source_geom.colors = dpcd2.colors
    vis.update_geometry(target_geom)
    vis.update_geometry(source_geom)
    print("Cores reais ativadas")
    return False

def stop_optimization(vis):
    shared['stop'] = True
    print("🚨 Otimização abortada pelo utilizador.")
    return False

# Teclas:
# C → muda modo de cor
# R → cores reais
# Q → aborta otimização
vis.register_key_callback(ord("C"), toggle_color)
vis.register_key_callback(ord("R"), use_real_colors)
vis.register_key_callback(ord("Q"), stop_optimization)

# ---------- INICIAR THREAD DE ICP ----------
initial_params = transformation_to_translation_rotation(trans_init)
opt_thread = threading.Thread(target=run_optimization, args=(initial_params, shared), daemon=True)
opt_thread.start()

# ---------- LOOP PRINCIPAL ----------
try:
    while opt_thread.is_alive():
        if shared['stop']:
            break

        latest = shared.get('latest_points', None)
        if latest is not None:
            np.asarray(source_geom.points)[:] = latest
            vis.update_geometry(source_geom)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)

    final = shared.get('latest_points', None)
    if final is not None:
        np.asarray(source_geom.points)[:] = final
        vis.update_geometry(source_geom)
        vis.poll_events()
        vis.update_renderer()

    print("Pressiona ESC ou fecha a janela para sair.")
    vis.run()
finally:
    vis.destroy_window()

# ---------- MOSTRAR RESULTADOS FINAIS ----------
if not shared['stop']:
    snapshots = shared['snapshots']
    if len(snapshots) > 0:
        print("A mostrar os principais passos da otimização...")
        entities = [copy.deepcopy(dpcd1), copy.deepcopy(dpcd2)]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for i, pts in enumerate(snapshots):
            pcd_step = o3d.geometry.PointCloud()
            pcd_step.points = o3d.utility.Vector3dVector(pts)
            pcd_step.paint_uniform_color(colors[i % len(colors)])
            entities.append(pcd_step)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        entities.append(axes)
        traj = view["trajectory"][0]
        o3d.visualization.draw_geometries(
            entities,
            front=traj["front"],
            lookat=traj["lookat"],
            up=traj["up"],
            zoom=traj["zoom"],
            window_name="Passos da Otimização ICP"
        )



