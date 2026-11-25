from glob import glob
from setuptools import find_packages, setup

setup(
    name='ijcas_experiments', version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/ijcas_experiments']),
        ('share/ijcas_experiments', ['package.xml']),
        ('share/ijcas_experiments/launch', glob('launch/*.launch.py')),
        ('share/ijcas_experiments/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Junggeun Park', maintainer_email='jk@example.com',
    description='IJCAS experiment reproduction interfaces', license='UNLICENSED',
    entry_points={'console_scripts':[
        'check = ijcas_experiments.cli:check_main',
        'reproduce_results = ijcas_experiments.cli:results_main',
        'sensor_bridge = ijcas_experiments.cli:sensor_main',
        'isaac = ijcas_experiments.cli:isaac_main',
        'experiment = ijcas_experiments.workflow:main',
    ]},
)
