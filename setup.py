from distutils.core import setup

setup(
    name='coalFLUT',
    version='1.1.0',
    packages=['coalFLUT', 'coalPFLUT', 'coal_calculation',
              'coalFLUT.diffusion', 'coalFLUT.premix'],
    scripts=['scripts/runCoalFLUT'],
    url='',
    license='',
    author='Michele Vascellari',
    author_email='Michele.Vascellari@vtc.tu-freiberg.de',
    description=('Generate FLUT for coal combustion using'
                 ' multi-dimensional flamelet'),

    install_requires=[
        'pytest',
        'termcolor',
        'autologging',
        # 'pyFLUT',
        'mock']
)
