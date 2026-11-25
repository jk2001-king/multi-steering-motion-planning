from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

data = [
    ('share/ament_index/resource_index/packages', ['resource/ijcas_experiments']),
    ('share/ijcas_experiments', ['package.xml', 'THIRD_PARTY_NOTICES.txt']),
]
for name in ('launch', 'config', 'runtime', 'assets', 'tools', 'docs'):
    for folder in [Path(name), *sorted(p for p in Path(name).rglob('*') if p.is_dir())]:
        files = sorted(str(p) for p in folder.glob('*') if p.is_file()
                       and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
        if files:
            data.append(('share/ijcas_experiments/' + str(folder), files))

setup(
    name='ijcas_experiments', version='0.1.0',
    packages=find_packages(exclude=['test']),
    py_modules=['ijcas_paths'], data_files=data,
    install_requires=['setuptools'], tests_require=['pytest'], zip_safe=False,
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
