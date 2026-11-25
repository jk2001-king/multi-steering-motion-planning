from setuptools import find_packages, setup

package_name = 'patient_initial_stability'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/'+package_name]),
        ('share/'+package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jk',
    maintainer_email='jk@example.com',
    description='Shared planning and response models for the IJCAS experiments.',
    license='Apache-2.0',
)
