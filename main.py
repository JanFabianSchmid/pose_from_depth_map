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
    color_image.save(output_dir / "output_color_image.png")

    points_world_frame = create_point_cloud_from_depth(depth_map, camera_matrix, extrinsics)
    pcl = create_point_cloud_object(points_world_frame=points_world_frame, rgb_image=rgb_image)
    o3d.io.write_point_cloud(output_dir / "output_point_cloud.ply", pcl)


if __name__ == "__main__":
    main()
