from setuptools import find_packages, setup

package_name = 'semantic_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/semantic_segmentation.launch.py']),
        ('share/' + package_name + '/config', [
            'config/config.yaml',
            'config/config_MIC.yaml',
            'config/config_STseg_front.yaml',
            'config/config_STseg_back.yaml',
            'config/config_STseg_left.yaml',
            'config/config_STseg_right.yaml',
        ])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hoons21',
    maintainer_email='hoons21@snu.ac.kr',
    description='Semantic segmentation package for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'semantic_segmentation = semantic_segmentation.semantic_segmentation:main',
            'semantic_segmentation_MIC = semantic_segmentation.semantic_segmentation_MIC:main',
            'semantic_segmentation_STseg_front = semantic_segmentation.semantic_segmentation_STseg_front:main',
            'semantic_segmentation_STseg_back = semantic_segmentation.semantic_segmentation_STseg_back:main',
            'semantic_segmentation_STseg_left = semantic_segmentation.semantic_segmentation_STseg_left:main',
            'semantic_segmentation_STseg_right = semantic_segmentation.semantic_segmentation_STseg_right:main',
        ],
    },
)
