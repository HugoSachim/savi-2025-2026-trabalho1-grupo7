#!/usr/bin/env python3

import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import copy
from copy import deepcopy
import pyvista as pv
import pyvistaqt as pvqt
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
import sys

# ----- Classe da interface -----
class ICPViewer(QMainWindow):
    def __init__(self, pcd1, pcd2, transforms, titles):
        super().__init__()
        self.setWindowTitle("ICP Registration Step Viewer (Open3D)")
        self.setGeometry(200, 100, 380, 350)

        self.pcd1 = pcd1
        self.pcd2 = pcd2
        self.transforms = transforms
        self.titles = titles
        self.color_mode = "real"  # pode ser: real, redblue, yellowgreen
        self.saved_view = None  # guarda a posição da câmara entre visualizações

        # Layout principal
        layout = QVBoxLayout()

        # Título
        label = QLabel("Visualizar etapas ICP")
        layout.addWidget(label)

        # Botões de etapas
        for i, title in enumerate(titles):
            btn = QPushButton(title)
            btn.clicked.connect(lambda _, step=i: self.show_step(step))
            layout.addWidget(btn)

        layout.addWidget(QLabel("Modo de cores"))
        # Botões de esquema de cores
        btn_real = QPushButton("Cores reais (RGB)")
        btn_real.clicked.connect(lambda: self.set_color_mode("real"))
        layout.addWidget(btn_real)

        btn_rb = QPushButton("Vermelho e Azul")
        btn_rb.clicked.connect(lambda: self.set_color_mode("redblue"))
        layout.addWidget(btn_rb)

        btn_yg = QPushButton("Amarelo e Verde")
        btn_yg.clicked.connect(lambda: self.set_color_mode("yellowgreen"))
        layout.addWidget(btn_yg)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ----- Função para mudar modo de cores -----
    def set_color_mode(self, mode):
        self.color_mode = mode
        print(f">> Modo de cor alterado para: {mode}")

    # ----- Função para mostrar etapa específica -----
    def show_step(self, step):
        transformation = self.transforms[step]
        title = self.titles[step]
        print(f">> A mostrar: {title}")

        # Copiar para não alterar os originais
        src = copy.deepcopy(self.pcd2)
        tgt = copy.deepcopy(self.pcd1)
        src.transform(transformation)

        # Aplicar esquema de cores
        if self.color_mode == "real":
            # Mantém cores RGB reais
            pass
        elif self.color_mode == "redblue":
            src.paint_uniform_color([1, 0, 0])   # vermelho
            tgt.paint_uniform_color([0, 0, 1])   # azul
        elif self.color_mode == "yellowgreen":
            src.paint_uniform_color([1, 1, 0])   # amarelo
            tgt.paint_uniform_color([0, 1, 0])   # verde

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # Criar visualizador interativo e manter câmara entre visualizações
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(src)
        vis.add_geometry(tgt)
        vis.add_geometry(axes)

        ctr = vis.get_view_control()
        if self.saved_view is not None:
            ctr.convert_from_pinhole_camera_parameters(self.saved_view)
        else:
            self.saved_view = ctr.convert_to_pinhole_camera_parameters()

        vis.run()
        # Guardar o estado da câmara atual
        self.saved_view = ctr.convert_to_pinhole_camera_parameters()
        vis.destroy_window()


# ----- Função principal -----
def main():
    # --- Carregar imagens RGB-D e gerar point clouds ---
    filename_rgb1 = "./tum_dataset/rgb/1.png"
    filename_depth1 = "./tum_dataset/depth/1.png"
    filename_rgb2 = "./tum_dataset/rgb/2.png"
    filename_depth2 = "./tum_dataset/depth/2.png"

    rgb1 = o3d.io.read_image(filename_rgb1)
    depth1 = o3d.io.read_image(filename_depth1)
    rgb2 = o3d.io.read_image(filename_rgb2)
    depth2 = o3d.io.read_image(filename_depth2)

    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    #print('1')
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    #print('2')
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intrinsics)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intrinsics)

    o3d.io.write_point_cloud(filename = "pointcloud1.ply", pointcloud = pcd1, 
                                format = 'auto', write_ascii = False, 
                                compressed = False, 
                                print_progress = False)

    o3d.io.write_point_cloud(filename = "pointcloud2.ply", pointcloud = pcd2, 
                                format = 'auto', write_ascii = False, 
                                compressed = False, 
                                print_progress = False)

    # --- Transformação inicial (obtida externamente) ---
    # trans_ext = np.asarray([
    #     [0.996857583523, -0.047511439770, -0.063385106623, 0.120132803917],
    #     [0.047460384667, 0.998870432377, -0.002311720978, 0.130328208208],
    #     [0.063423343003, -0.000703825208, 0.997986435890, 0.168358176947],
    #     [0, 0, 0, 1]

    trans_ext = np.asarray([[0.983646262139, -0.081055920198, 0.160841439876, -0.901536037487],
                         [0.079983057301, 0.996709553274, 0.013144464908, -0.001155507942],
                         [-0.161377636385, -0.000064913673, 0.986892726825, 0.136104476608], 
                         [0.0, 0.0, 0.0, 1.0]])

    # --- Fast Global Registration ---
    def preprocess_point_cloud(pcd, voxel_size, max_neighbors):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=max_neighbors))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=max_neighbors))
        pcd_normals = pcd_down
        pcd_normals.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=max_neighbors))
        pcd_normals.orient_normals_consistent_tangent_plane(5)
        return pcd_down, pcd_fpfh, pcd_normals
    #print('3')
    voxel_size = 0.025
    max_neighbors = 20
    print('A calcular downsampled, normals e características FPFH')
    source_down, source_fpfh, source_normals = preprocess_point_cloud(pcd2, voxel_size, max_neighbors)
    target_down, target_fpfh, target_normals = preprocess_point_cloud(pcd1, voxel_size, max_neighbors)
    print('downsampled from' + str(pcd2) + ' to ' + str(source_down))
    print('downsampled from' + str(pcd1) + ' to ' + str(target_down))


    result_fast = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size)
    )
    trans_registration = result_fast.transformation
    # trans_registration = trans_ext
    #print('4')
    print("Transformação inicial:")
    print(trans_registration)

    # --- ICP refinamento ---
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, trans_ext,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    trans_icp = reg_p2p.transformation

    print("Transformação final (ICP):")
    print(trans_icp)

    # --- ICP refinamento point to plane ---
    reg_p2plane = o3d.pipelines.registration.registration_icp(
        source_normals, target_normals, threshold, trans_ext,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    trans_icp_plane = reg_p2plane.transformation

    print("Transformação final (ICP plane):")
    print(trans_icp_plane)

    # --- Iniciar GUI ---
    app = QApplication(sys.argv)
    transforms = [
        np.eye(4),
        trans_ext,
        trans_registration,
        trans_icp,
        trans_icp_plane 
    ]
    titles = [
        "Nuvens Iniciais",
        "Transformação Inicial",
        "Fast Global Registration",
        "Resultado ICP",
        "Resultado ICP Plane"
    ]

    viewer = ICPViewer(pcd1, pcd2, transforms, titles)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


   