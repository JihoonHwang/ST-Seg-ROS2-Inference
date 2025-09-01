#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _launch_nodes(context, *args, **kwargs):
    versions = LaunchConfiguration('versions').perform(context)
    pkg_share = get_package_share_directory('semantic_segmentation')
    
    nodes = []
    
    # 쉼표로 구분된 버전들을 파싱
    version_list = [v.strip() for v in versions.split(',')]
    
    for version in version_list:
        if version == 'MIC':
            exe_name    = f'semantic_segmentation_{version}'
            params_file = os.path.join(pkg_share, 'config', f'config_{version}.yaml')
            node_name   = f'semantic_segmentation_{version}'
        elif version in ['STseg_front', 'STseg_back', 'STseg_left', 'STseg_right']:
            exe_name    = f'semantic_segmentation_{version}'
            params_file = os.path.join(pkg_share, 'config', f'config_{version}.yaml')
            node_name   = f'semantic_segmentation_{version}'
        elif version in ['semantic', 'base']:
            exe_name    = 'semantic_segmentation'
            params_file = os.path.join(pkg_share, 'config', 'config.yaml')
            node_name   = 'semantic_segmentation'
        else:
            print(f"Warning: Unknown version '{version}', skipping...")
            continue
            
        node = Node(
            package='semantic_segmentation',
            executable=exe_name,
            name=node_name,
            output='screen',
            parameters=[params_file],
        )
        nodes.append(node)
    
    return nodes

def generate_launch_description():
    versions_arg = DeclareLaunchArgument(
        'versions',
        default_value='semantic',
        description='Comma-separated list of versions to run. Options: semantic, MIC, STseg_front, STseg_back, STseg_left, STseg_right'
    )

    return LaunchDescription([
        versions_arg,
        OpaqueFunction(function=_launch_nodes),
    ])