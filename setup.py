from distutils.core import setup

setup(
    name='coalFLUT',
    version='1.0.0',
    packages=['coalFLUT', 'coal_calculation'],
    scripts=['scripts/runCoalFLUT'],
    url='',
    license='',
    author='Michele Vascellari',
    author_email='Michele.Vascellari@vtc.tu-freiberg.de',
    description=('Generate FLUT for coal combustion using'
                 ' multi-dimensional flamelet')
)
