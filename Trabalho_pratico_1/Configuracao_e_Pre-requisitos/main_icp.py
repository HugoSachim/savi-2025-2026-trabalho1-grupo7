#!/usr/bin/env python3
# shebang line for linux / mac

import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import copy
from copy import deepcopy
import time

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
voxel_size = 0.01 #0.005 meters
#downsampled_point_cloud
dpcd1 = deepcopy(pcd1)
dpcd2 = deepcopy(pcd2)
dpcd1 = dpcd1.voxel_down_sample(voxel_size)
dpcd2 = dpcd2.voxel_down_sample(voxel_size)
print('downsampled_point_cloud 1 from' + str(pcd1) + 'to ' + str(dpcd1))
print('downsampled_point_cloud 2 from' + str(pcd2) + 'to ' + str(dpcd2))

draw_geometries(dpcd1,dpcd2, view, False, "Downsampling")



# Inicial transformation matrix obtained externally

trans_init = np.asarray([[0.996857583523, -0.047511439770, -0.063385106623, 0.120132803917],
                        [0.047460384667, 0.998870432377, -0.002311720978, 0.130328208208],
                        [0.063423343003, -0.000703825208, 0.997986435890, 0.168358176947], [0, 0, 0, 1]])

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

# draw_registration_result(pcd2, pcd1, trans_init)

# ---------- Point to Point ICP -----------------------
# The point cloud source is the one transformed by the calculated trasform matrix

max_correspondence_distance = 0.1

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    dpcd2,
    dpcd1,
    max_correspondence_distance,
    init=trans_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(dpcd2, dpcd1, reg_p2p.transformation, view, False, 'ICP downsampling point to point')
# show with all the points

def estimate_normals(pcd):
    pcd_normal = deepcopy(pcd)
    pcd_normal.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01,   # raio de procura dos vizinhos (em metros)
            max_nn=5     # máximo de vizinhos a considerar
        )
    )
    pcd_normal.orient_normals_consistent_tangent_plane(5)
    return pcd_normal

dpcd2_normal = estimate_normals(dpcd2)
dpcd1_normal = estimate_normals(dpcd1)

#draw_registration_result(dpcd2_normal, dpcd1_normal, reg_p2p.transformation, view, False, 'ICP downsampling with normals')

draw_registration_result(pcd2, pcd1, reg_p2p.transformation, view, False, 'ICP point to point')

# pcd2_normal = estimate_normals(pcd2)
# pcd1_normal = estimate_normals(pcd1)
#draw_registration_result(pcd2_normal, pcd1_normal, reg_p2p.transformation, view, False, 'ICP with normals')

# Default: fitness = 0,912 inlier = 0,0736
# 100 iter: fitness = 0,936 inlier = 0,0490

print("Apply point-to-plane ICP")
reg_p2plane = o3d.pipelines.registration.registration_icp(
    dpcd2_normal, 
    dpcd1_normal, 
    max_correspondence_distance, 
    init=trans_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
)
print(reg_p2plane)
print("Transformation is:")
print(reg_p2plane.transformation)
draw_registration_result(dpcd2_normal, dpcd1_normal, reg_p2plane.transformation, view, False, 'ICP downsampling point to plane')
# show with all the points

draw_registration_result(pcd2, pcd1, reg_p2plane.transformation, view, False, 'ICP point to plane')

