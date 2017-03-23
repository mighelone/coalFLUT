from distutils.core import setup

setup(
    name='coalFLUT',
    version='1.0.0',
    packages=['coalFLUT', 'coalPFLUT', 'coal_calculation'],
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
        'pyFLUT>2.4.0',
        'mock']
)
