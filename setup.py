from setuptools import setup

# adapted from https://github.com/f1tenth/f1tenth_gym

setup(name='f110_gym',
      version='0.2.1',
      package_dir={'': 'gym'},
      install_requires=['gym==0.19.0',
		        'numpy<=1.22.0,>=1.18.0',
                        'Pillow>=9.0.1',
                        'scipy>=1.7.3',
                        'numba>=0.55.2',
                        'pyyaml>=5.3.1',
                        'pyglet<1.5',
                        'torch~=2.1.2',
                        'tqdm~=4.66.1',
                        'casadi~=3.6.4',
                        'matplotlib~=3.8.2',
                        'torchvision~=0.16.2',
                        'tensorboard~=2.15.1',
                        'configargparse~=1.7',
                        'cvxpy~=1.4.1',
                        'termcolor~=2.4.0']
      )
