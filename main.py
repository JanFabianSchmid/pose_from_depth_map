import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def decode_rgb444(color):
    # Decode the color map as an RGB444 image
    # Assuming the color map is encoded as 12-bit RGB444
    r = (((color.astype(np.uint16)) >> 8) & 0xF) * 17  # Extract the red channel and scale to 0-255
    g = (((color.astype(np.uint16)) >> 4) & 0xF) * 17  # Extract the green channel and scale to 0-255
    b = ((color.astype(np.uint16)) & 0xF) * 17  # Extract the blue channel and scale to 0-255

    # Stack the channels to form an RGB image
    return np.stack((r, g, b), axis=-1).astype(np.uint8)


def create_point_cloud_from_depth(depth, camera_matrix, extrinsics):
    height, width = depth.shape
    # Create a point cloud from the depth map
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    coords_2d_homogeneous = np.vstack((i.ravel(), j.ravel(), np.ones(i.size)))

    # Back-project to 3D space
    depth_flat = depth.ravel()
    # Project from image coordinates to 3D points
    points_camera_frame = np.linalg.inv(camera_matrix) @ (coords_2d_homogeneous * depth_flat)

    # Transform points to world frame using extrinsics
    points_camera_frame_homogeneous = np.vstack((points_camera_frame, np.ones((1, points_camera_frame.shape[1]))))
    points_world_frame = extrinsics @ points_camera_frame_homogeneous

    return points_world_frame


def create_point_cloud_object(points_world_frame, rgb_image):
    # Extract 3D points
    point_cloud = points_world_frame[:3, :].T

    # Flatten the RGB image to match the point cloud
    colors_flat = rgb_image.reshape(-1, 3)

    # Create an Open3D PointCloud object
    pcl = o3d.geometry.PointCloud()

    # Set the points and colors
    pcl.points = o3d.utility.Vector3dVector(point_cloud)
    pcl.colors = o3d.utility.Vector3dVector(colors_flat / 255.0)  # Normalize colors to [0, 1]

    return pcl


def plane_segmentation(pcl, plane_dist_thresh=0.01):
    _, inliers = pcl.segment_plane(distance_threshold=plane_dist_thresh, ransac_n=3, num_iterations=1000)
    plane_cloud = pcl.select_by_index(inliers)
    object_cloud = pcl.select_by_index(inliers, invert=True)
    return object_cloud, plane_cloud


def detect_clusters(pcl, cluster_eps=0.02, min_points=10):
    # We use the DBSCAN algorithm to identify point clusters
    labels = np.array(pcl.cluster_dbscan(eps=cluster_eps, min_points=min_points))
    print(f"Detected {labels.max() + 1} clusters")
    return labels


def assign_cluster_colors(pcl, labels):
    # Assign colors to points based on their cluster label
    colors = np.zeros((len(labels), 3))
    for i in range(labels.max() + 1):
        color = np.random.rand(3)  # Generate a random color
        colors[labels == i] = color

    # Set the colors for the rest_cloud
    pcl.colors = o3d.utility.Vector3dVector(colors)
    return pcl


def find_closest_planes(object_cloud, cluster_labels):
    # Detect clusters that can be described as a plane with surface normal facing the camera and identify the closest and second closest
    closest_plane_id = None
    second_closest_plane_id = None
    max_z_coord = float("-inf")
    second_max_z_coord = float("-inf")
    for cluster_id in range(cluster_labels.max() + 1):
        cluster_points = object_cloud.select_by_index(np.where(cluster_labels == cluster_id)[0])
        plane_model, inliers = cluster_points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        if len(inliers) > 0.5 * len(cluster_points.points):  # Check if most points fit a plane
            normal = plane_model[:3]  # Extract the normal vector
            if np.allclose(normal, [0, 0, 1], atol=0.01):  # Check if the normal is approximately in the z-direction
                z_coords = np.asarray(cluster_points.points)[:, 2]  # Extract z-coordinates (distance along z-axis)
                avg_z_coord = np.mean(z_coords)
                if avg_z_coord > max_z_coord:
                    second_max_z_coord = max_z_coord
                    second_closest_plane_id = closest_plane_id
                    max_z_coord = avg_z_coord
                    closest_plane_id = cluster_id
                elif avg_z_coord > second_max_z_coord:
                    second_max_z_coord = avg_z_coord
                    second_closest_plane_id = cluster_id
    return closest_plane_id, second_closest_plane_id, max_z_coord, second_max_z_coord


def main():
    # Load the 4x4 extrinsics
    extrinsics = np.load("data/extrinsics.npy")
    print(extrinsics)
    # Load the 3x3 pinhole camera matrix
    camera_matrix = np.load("data/intrinsics.npy")
    print(camera_matrix)
    depth_map = np.load("data/one-box.depth.npdata.npy")
    # Depth shape is 1544x2064
    color_map = np.load("data/one-box.color.npdata.npy")
    # Color shape is 1544x2064

    # Create the output directory if it does not exist
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_image = decode_rgb444(color_map)
    color_image = Image.fromarray(rgb_image)
    color_image.save(output_dir / "color_image.png")

    points_world_frame = create_point_cloud_from_depth(depth_map, camera_matrix, extrinsics)
    pcl = create_point_cloud_object(points_world_frame=points_world_frame, rgb_image=rgb_image)
    o3d.io.write_point_cloud(output_dir / "point_cloud.ply", pcl)

    # Downsample
    voxel_size: float = 0.01
    pcl = pcl.voxel_down_sample(voxel_size)

    # Remove outliers
    pcl, _ = pcl.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Remove ground plane from the rest (i.e. objects)
    object_cloud, plane_cloud = plane_segmentation(pcl, plane_dist_thresh=0.01)

    cluster_labels = detect_clusters(object_cloud)
    o3d.io.write_point_cloud(
        output_dir / "point_cloud_clustered.ply", assign_cluster_colors(object_cloud, cluster_labels)
    )

    closest_plane_id, second_closest_plane_id, max_z_coord, second_max_z_coord = find_closest_planes(
        object_cloud, cluster_labels
    )

    if closest_plane_id is not None and second_closest_plane_id is not None:
        z_distance = max_z_coord - second_max_z_coord

    # Fit a square into the closest_plane
    if closest_plane_id is not None:
        closest_plane_points = object_cloud.select_by_index(np.where(cluster_labels == closest_plane_id)[0])

        # Fit an oriented bounding box to the closest plane points
        obb = closest_plane_points.get_oriented_bounding_box()

        # Set the z-length of the bounding box to z_distance
        extent = np.array(obb.extent)
        extent[2] = z_distance
        new_center = obb.center - np.array([0, 0, z_distance / 2])  # Lower the center by z_distance/2 along the z-axis
        obb = o3d.geometry.OrientedBoundingBox(new_center, obb.R, extent)

        # Save the oriented bounding box to disk
        bounding_box_data = {
            "center": obb.center.tolist(),
            "extent": obb.extent.tolist(),
            "rotation_matrix": obb.R.tolist(),
        }
        with open(output_dir / "bounding_box.json", "w") as f:
            json.dump(bounding_box_data, f, indent=4)


if __name__ == "__main__":
    main()
