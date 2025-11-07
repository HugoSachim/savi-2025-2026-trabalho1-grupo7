import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import copy
from copy import deepcopy

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [10.0, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.87911045824568079, -0.1143707949631662, 0.46269225567601935],
                "lookat": [-14.857198715209961, 8.7558956146240234, 4.6393190026283264],
                "up": [-0.45122740480118839, 0.11291073802962912, 0.88523725316662361],
                "zoom": 0.53999999999999981
            }
        ],
    "version_major": 1,
    "version_minor": 0
}

# Open RGB file 1
filename_rgb1 = "SAVI_Mateus/Aula08/tum_dataset/rgb/1.png"
rgb1 = o3d.io.read_image(filename_rgb1)

# Open depth file 1
filename_depth1 = "SAVI_Mateus/Aula08/tum_dataset/depth/1.png"
depth1 = o3d.io.read_image(filename_depth1)

# Create the rgbd image for file 1
rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
print(rgbd1)

# Open RGB file 2
filename_rgb2 = 'SAVI_Mateus/Aula08/tum_dataset/rgb/2.png'
rgb2 = o3d.io.read_image(filename_rgb2)

# Open depth flie 2
filename_depth2 = 'SAVI_Mateus/Aula08/tum_dataset/depth/2.png'
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

axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)

# Save the point cloud generated to the program folder

o3d.io.write_point_cloud(filename = "SAVI_Mateus/Aula08/tum_dataset/pointcloud1.ply", pointcloud = pcd1, 
                            format = 'auto', write_ascii = False, 
                            compressed = False, 
                            print_progress = False)

o3d.io.write_point_cloud(filename = "SAVI_Mateus/Aula08/tum_dataset/pointcloud2.ply", pointcloud = pcd2, 
                            format = 'auto', write_ascii = False, 
                            compressed = False, 
                            print_progress = False)


#----------------- Display raw point clouds with different colors ----------#

# # paint points to get a better visualization
# pcd1.paint_uniform_color([1, 0, 0])  # reg, green, blue
# pcd2.paint_uniform_color([0, 0, 1])
# entities = [pcd1, pcd2, axes_mesh]

# # Draw the geometries
# o3d.visualization.draw_geometries(entities,
#                                 front=view['trajectory'][0]['front'],
#                                 lookat=view['trajectory'][0]['lookat'],
#                                 up=view['trajectory'][0]['up'],
#                                 zoom=view['trajectory'][0]['zoom'],)

#---------------------------------------------------------------------------#

# Inicial transformation matrix obtained externally

trans_ext = np.asarray([[0.996857583523, -0.047511439770, -0.063385106623, 0.120132803917],
                        [0.047460384667, 0.998870432377, -0.002311720978, 0.130328208208],
                        [0.063423343003, -0.000703825208, 0.997986435890, 0.168358176947], [0, 0, 0, 1]])

# Function to draw both point clouds transformed with transformation matrix

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

#Show the 2 Point Clouds with the inicial external transformation   

# draw_registration_result(pcd2, pcd1, trans_ext)

#---------------------- Fast Global Registration ---------------------------------#

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = pcd2
    target = pcd1
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

voxel_size = 0.005  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size)

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

result_fast = execute_fast_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh,
                                               voxel_size)

print(result_fast)
draw_registration_result(source_down, target_down, result_fast.transformation)

trans_registration = result_fast.transformation

#---------------------------------------------------------------------------------#


# ----------------------- Point to Point ICP -------------------------------------#
# The point cloud source is the one transformed by the calculated trasform matrix

threshold = 0.1

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(pcd2, pcd1, threshold, trans_registration,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(pcd2, pcd1, reg_p2p.transformation)

# Default: fitness = 0,912 inlier = 0,0736
# 100 iter: fitness = 0,936 inlier = 0,0490