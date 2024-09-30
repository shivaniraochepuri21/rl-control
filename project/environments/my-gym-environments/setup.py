from setuptools import setup, find_packages

setup(
    name='my_gym_environments',
    version='0.0.1',
    description='Multiple Custom Gymnasium environments',
    author='Shivani Chepuri',
    author_email='shivanirao.chepuri21@gmail.com',
    # url='https://github.com/yourusername/custom_gym_env',  # Optional
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium==0.29.0',
    ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
#    entry_points={
#        'gymnasium.envs': [
#            'LinearFirstOrderEnv = my_gym_environments:LinearFirstOrderEnv',
#            'linearsecondorder = my_gym_environments:LinearSecondOrderEnv',
#            'my_pendulum = my_gym_environments:MyPendulumEnv',
#        ],
#    },
)


