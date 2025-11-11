#!/usr/bin/env python3
# shebang line for linux / mac

import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import copy
from copy import deepcopy
import time
import scipy
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from functools import partial

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.1947980721791587, 0.98024508257841203, 4.4420065482432269 ],
			"boundingbox_min" : [ -2.5292977056043364, -1.1010999520619711, 1.4601356643334618 ],
			"field_of_view" : 60.0,
			"front" : [ -0.44990825844154053, 0.42727035899712068, -0.78423376572841585 ],
			"lookat" : [ -1.5100770025260728, 0.28706664764620715, 1.6367510876634088 ],
			"up" : [ 0.0037591866583305822, -0.87721395852113138, -0.48008492945660647 ],
			"zoom" : 0.34589999999999987
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

# Open RGB file 1
filename_rgb1 = "./tum_dataset/rgb/1.png"
rgb1 = o3d.io.read_image(filename_rgb1)

# Open depth file 1
filename_depth1 = "./tum_dataset/depth/1.png"
depth1 = o3d.io.read_image(filename_depth1)

# Create the rgbd image for file 1
rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
print(rgbd1)

# Open RGB file 2
filename_rgb2 = './tum_dataset/rgb/2.png'
rgb2 = o3d.io.read_image(filename_rgb2)

# Open depth flie 2
filename_depth2 = './tum_dataset/depth/2.png'
depth2 = o3d.io.read_image(filename_depth2)

# Create the rgbd image for file 2
rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
print(rgbd2)

# Obtain the point cloud from the rgbd images
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd1, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd2, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Save the point cloud generated to the program folder

o3d.io.write_point_cloud(filename = "pointcloud1.ply", pointcloud = pcd1, 
                            format = 'auto', write_ascii = False, 
                            compressed = False, 
                            print_progress = False)

o3d.io.write_point_cloud(filename = "pointcloud2.ply", pointcloud = pcd2, 
                            format = 'auto', write_ascii = False, 
                            compressed = False, 
                            print_progress = False)

# # Draw the original geometries
def draw_geometries(pcd1,pcd2,view ,paint, name):
    #paint False for original color, True for painted
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    pcd1_paint= deepcopy(pcd1)
    pcd2_paint= deepcopy(pcd2)
    if paint:
        pcd1_paint.paint_uniform_color([0, 0.651, 0.929])  # reg, green, blue
        pcd2_paint.paint_uniform_color([1, 0.706, 0])
    entities = [pcd1_paint, pcd2_paint, axes_mesh]

    o3d.visualization.draw_geometries(entities,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'],
                                        zoom=view['trajectory'][0]['zoom'],
                                        window_name=name)

draw_geometries(pcd1,pcd2, view, False, "Original")

# # Downsampling
voxel_size = 0.025 #0.025 meters
#downsampled_point_cloud
dpcd1 = deepcopy(pcd1)
dpcd2 = deepcopy(pcd2)
dpcd1 = dpcd1.voxel_down_sample(voxel_size)
dpcd2 = dpcd2.voxel_down_sample(voxel_size)
print('downsampled_point_cloud 1 from' + str(pcd1) + 'to ' + str(dpcd1))
print('downsampled_point_cloud 2 from' + str(pcd2) + 'to ' + str(dpcd2))

draw_geometries(dpcd1,dpcd2, view, False, "Downsampling")



# Inicial transformation matrix obtained externally
trans_init = np.asarray([[0.983646262139, -0.081055920198, 0.160841439876, -0.901536037487],
                         [0.079983057301, 0.996709553274, 0.013144464908, -0.001155507942],
                         [-0.161377636385, -0.000064913673, 0.986892726825, 0.136104476608], 
                         [0.0, 0.0, 0.0, 1.0]])

# Function to draw both point clouds 

def draw_registration_result(source, target, transformation, view, paint, name):
    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    if paint:
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # reg, green, blue
        source_temp.paint_uniform_color([1, 0.706, 0])
    entities = [source_temp, target_temp, axes_mesh]
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      window_name=name)

# Show the 2 Point Clouds with the inicial external transformation   
# Apply transformation to point clound
def apply_transformation(pcd, trans):
    pcd_transformed = copy.deepcopy(pcd)
    pcd_transformed.transform(trans)
    return pcd_transformed

pcd2_trans = apply_transformation(pcd2, trans_init)
draw_geometries(pcd1,pcd2_trans, view, False, "Inicial transformation")

# ---------- Point to Point ICP -----------------------
# The point cloud source is the one transformed by the calculated trasform matrix


def translation_rotation_to_transformation(rot_trans):
    """
    rot_trans: [rx, ry, rz, tx, ty, tz]
    -> 4x4 matriz homogénea

    R.from_rotvec(xi[:3]) cria uma rotação a partir de um vetor rotação (axis-angle).
    A direção indica o eixo. euler
    O comprimento é o ângulo (em radianos).
    as_matrix() converte para matriz 3×3.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rot_trans[:3]).as_matrix()
    T[:3, 3] = rot_trans[3:]
    return T

#print(translation_rotation_to_transformation([0.5,0.4,3.0,2,3,6]))

def transformation_to_translation_rotation(trans):
    assert trans.shape == (4, 4), "A matriz deve ser 4x4"
    R_mat = trans[:3, :3]
    t = trans[:3, 3]
    # Converte matriz de rotação em vetor rotação (axis-angle)
    rot = Rotation.from_matrix(R_mat)
    rotvec = rot.as_rotvec()
    return np.concatenate([rotvec, t])

def error_target_source(pcd_source, pcd_target, keep_ratio=0.90): #ratio para tirar outliners
    # start = time.time() 
    kdtree = o3d.geometry.KDTreeFlann(pcd_target)
    # end = time.time()
    # elapsed = end - start
    # print(f"Execution time kd tree: {elapsed:.6f} seconds")

    src_pts = np.asarray(pcd_source.points)
    tgt_pts = np.asarray(pcd_target.points)

    residuals = []

    # start1 = time.time() 
    for p in src_pts:
        [_, idx, _] = kdtree.search_knn_vector_3d(p, 1)
        tgt_p = tgt_pts[idx[0]]
        diff = p - tgt_p
        total_squared_distance = np.sum(diff**2) #distancia euclidiana
        residuals.append(total_squared_distance)
    # end1 = time.time()
    # elapsed1 = end1 - start1
    # print(f"Execution time loop for: {elapsed1:.6f} seconds")

    residuals = np.array(residuals)

    # # 🔹 Keep only the lowest X% of distances (best matches)
    # n_keep = int(len(residuals) * keep_ratio)
    # residuals = np.sort(residuals)[:n_keep]

    n_keep = int(len(residuals) * keep_ratio)
    threshold = np.partition(residuals, n_keep-1)[n_keep-1]  # find cutoff
    residuals = residuals[residuals <= threshold]

    # n_keep = int(len(residuals) * keep_ratio)
    # residuals = residuals[:n_keep]

    return residuals

def view_trans(pcd_source, pcd_target, view, paint):
    traj = view["trajectory"][0]
    front  = traj["front"]
    lookat = traj["lookat"]
    up     = traj["up"]
    zoom   = traj["zoom"]
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualização rápida")

    if paint:
        pcd_target.paint_uniform_color([0, 0.651, 0.929])  # reg, green, blue
        pcd_source.paint_uniform_color([1, 0.706, 0])

    vis.add_geometry(pcd_target)   # primeiro
    vis.add_geometry(pcd_source)   # segundo

    # Configurar câmera
    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    # Mostrar por 2 segundos
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.25)

    # Fechar
    vis.destroy_window()

# Define the objective function
def objectiveFunction(params, shared_mem):
    # Extract the parameters
    rx, ry, rz, tx, ty, tz = params
    print('Transformation = ' + str(params))
    trans = translation_rotation_to_transformation(params)
    #print('Transformation = ' + str(trans))

    pcd_target = shared_mem['pcd_target']
    pcd_source = shared_mem['pcd_source']
    view = shared_mem['view']

    # Applying transformation
    pcd_source = apply_transformation(pcd_source, trans)

    # Compute the error
    error = error_target_source(pcd_source, pcd_target)
    total_squared_error = np.sum(error**2)   
    print('Error = ' + str(total_squared_error))

    #view_trans(pcd_source, pcd_target, view, False)

    #print('one more iteration')
    return error


# # Start optimization
shared_mem = {'pcd_target': dpcd1, 'pcd_source': dpcd2, 'view': view}

initial_params = transformation_to_translation_rotation(trans_init)
#print(initial_params)
#initial_params = [-0.00664098, 0.16199477, 0.1, -0.90153604, -0.003, 0.3]
#initial_params = [0.0, 0.02, 0.0, 1.0, 1.0, 1.0]


#error = objectiveFunction(initial_params, shared_mem)

result = least_squares(partial(objectiveFunction, shared_mem=shared_mem),
                        initial_params)
# result = least_squares(partial(objectiveFunction, shared_mem=shared_mem),
#                         initial_params, diff_step=1.5)
#0.005
print('Optimization finished. Result=\n' + str(result))

#falta limitar o numero de iteracoes

params_opti = result.x.tolist()
trans_opti = translation_rotation_to_transformation(params_opti)
pcd_source_opti = apply_transformation(pcd2,trans_opti)
draw_geometries(pcd1,pcd_source_opti, view, False, "Final optimization")





