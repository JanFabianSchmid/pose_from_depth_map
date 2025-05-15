import json

import numpy as np
import open3d as o3d


def main():
    full_point_cloud = o3d.io.read_point_cloud("output/full_point_cloud.ply")
    o3d.visualization.draw_geometries([full_point_cloud])

    extracted_plane_cloud = o3d.io.read_point_cloud("output/extracted_plane.ply")
    o3d.visualization.draw_geometries([extracted_plane_cloud])

    full_point_cloud_clustered = o3d.io.read_point_cloud("output/full_point_cloud_clustered.ply")
    o3d.visualization.draw_geometries([full_point_cloud_clustered])

    # Read the bounding box data from the JSON file
    with open("output/bounding_box.json", "r") as f:
        bounding_box_data = json.load(f)

    # Recreate the oriented bounding box from the JSON data
    center = np.array(bounding_box_data["center"])
    extent = np.array(bounding_box_data["extent"])
    rotation_matrix = np.array(bounding_box_data["rotation_matrix"])
    obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)

    # Visualize the recreated bounding box
    obb.color = (0, 1, 0)  # Set the bounding box color to green

    # Create a mesh box from the oriented bounding box
    obb_mesh = o3d.geometry.TriangleMesh.create_box(*obb.extent)
    obb_mesh.translate(obb.center - obb.extent / 2)
    obb_mesh.rotate(obb.R, center=obb.center)

    # Set the color of the mesh box
    obb_mesh.paint_uniform_color([0, 1, 0])  # Green color for the surfaces

    # Add the mesh box to the visualization
    o3d.visualization.draw_geometries([full_point_cloud, obb_mesh])
