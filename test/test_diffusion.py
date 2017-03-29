import pytest
import numpy as np
import mock
import pyFLUT
import pyFLUT.ulf

from coalFLUT.diffusion import CoalDiffusionFLUT
import os
import shutil
import glob


ulf_file = os.path.abspath('examples/Le1/example.ulf')

settings = {'flut': {'c': 101,
                     'pv': {'CO': 1, 'CO2': 1},
                     'solver': 'OpenFOAM',
                     'varZ': {'method': 'linspace', 'values': [0, 1, 3]},
                     'variables': ['rho', 'T']},
            'oxidizer': {'T': {'max': 500, 'min': 300}, 'X': {'N2': 0.79, 'O2': 0.21}},
            'parameters': {'Hnorm': {'method': 'linspace', 'values': [0, 1, 2]},
                           'Y': {'method': 'linspace', 'values': [0, 1, 3]},
                           'chist': {'method': 'logspace', 'values': [-1, 1, 10]}},
            'pressure': 101325,
            'ulf': {'basename': 'test',
                    'keys': {'Zst': 'ZST',
                             'basename': 'RESULT_BASENAME',
                             'chist': 'CHIST',
                             'mechanism': 'MECHANISM',
                             'n_grid': 'REFINE_TARGET',
                             'pressure': 'PRESSURE'},
                    'mechanism': 'ch4_smooke.xml',
                    'n_grid': 101,
                    'par_format': '{:5.4f}',
                    'restart': True,
                    'solver': '/shared_home/vascella/Codes/ulf2/build/ulfrun',
                    'ulf_reference': 'flamelet_setup.ulf'},
            'volatiles': {'T': {'max': 600, 'min': 300},
                          'Y': {'CH4': 0.5, 'H2': 0.2, 'N2': 0}}}


@pytest.fixture(scope='module')
def test_path():
    """Setup the path for testing CoalDiffusionFLUT"""
    cwd = os.getcwd()
    path = os.path.abspath(os.path.join(cwd, 'test/test_diffusion'))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.chdir(path)
    CoalDiffusionFLUT.copy_files()
    yield path

    # teardown
    os.chdir(cwd)
    shutil.rmtree(path)


def ulf_run(self):
    """Mock function for UlfRun.run"""
    res_file = self['RESULT_BASENAME'] + 'final.ulf'
    shutil.copy(ulf_file, res_file)
    return pyFLUT.Flame1D.read_ulf(res_file)


@pytest.fixture
def flut(test_path):
    """Init CoalDiffusionFLUT"""
    cflut = CoalDiffusionFLUT(settings)
    yield cflut

    # teardown
    # remove intermediate files
    [os.remove(f) for f in glob.iglob('inp_*.ulf')]
    [os.remove(f)
     for f in glob.iglob('{}_*.ulf'.format(cflut.basename))]


def test_read(flut):
    assert hasattr(flut, 'chargases')
    np.testing.assert_allclose(flut.Y, np.linspace(0, 1, 3))
    np.testing.assert_allclose(flut.chist, np.logspace(-1, 1, 10))
    np.testing.assert_allclose(flut.Hnorm, np.linspace(0, 1, 2))
    np.testing.assert_allclose(flut.volatiles['T'], [300, 600])
    np.testing.assert_allclose(flut.oxidizer['T'], [300, 500])
    np.testing.assert_allclose(flut.chargases['T'], [300, 600])
    assert 'Hnorm' in flut._parameter
    assert 'Y' in flut._parameter
    assert 'chist' in flut._parameter


def test_chargases(flut):
    gas = flut.gas
    Mc = gas.atomic_weight('C')
    Mo2 = gas.molecular_weights[gas.species_index('O2')]
    Mn2 = gas.molecular_weights[gas.species_index('N2')]
    Mco = gas.molecular_weights[gas.species_index('CO')]

    n2_to_o2 = 79. / 21.
    alpha = 0.5 * (Mo2 + n2_to_o2 * Mn2) / Mc

    gas.TPY = 300, None, flut.oxidizer['Y']
    alpha_danny = 0.5 * gas.mean_molecular_weight / 0.21 / Mc
    assert alpha == flut.alphac
    assert alpha_danny == flut.alphac

    mass_tot = Mco + 0.5 * n2_to_o2 * Mn2
    assert flut.chargases['Y']['CO'] == Mco / mass_tot
    assert flut.chargases['Y']['N2'] == 0.5 * n2_to_o2 * Mn2 / mass_tot


@mock.patch('pyFLUT.ulf.UlfRun.run', new=ulf_run)
def test_run(flut):
    flut.run()
    assert 'Z' in flut
    for v in ['Z', 'Hnorm', 'Y', 'chist']:
        assert v in flut.input_variables

    assert hasattr(flut, 'basename')
    [os.remove(f) for f in glob.iglob(flut.basename + '_Y*.ulf')]
    [os.remove(f) for f in glob.iglob('inp_Y*.ulf')]


def calc_mix(p, v, Y, alphac):
    alphac_one = 1 + alphac
    return ((alphac_one * (1 - Y) * p + Y * v) /
            (alphac_one * (1 - Y) + Y))


def test_mix_fuels(flut):
    Y, Hnorm = 0, 0
    mix = flut.mix_fuels(Y, Hnorm, flut.gas)
    np.testing.assert_almost_equal(mix['T'], flut.chargases['T'][0])

    Y, Hnorm = 1, 1
    mix = flut.mix_fuels(Y, Hnorm, flut.gas)
    np.testing.assert_almost_equal(mix['T'], flut.volatiles['T'][1])

    Y, Hnorm = 0.5, 0.5
    mix = flut.mix_fuels(Y, Hnorm, flut.gas)

    CH4 = calc_mix(0, flut.volatiles['Y']['CH4'], Y, flut.alphac)
    np.testing.assert_almost_equal(CH4,
                                   mix['Y']['CH4'])
