import pytest
import os
import shutil
import glob
import numpy as np

from coalFLUT.premix import CoalPremixFLUT
from pyFLUT.ulf import UlfRun
import pyFLUT

ulf_results = os.path.abspath(
    'test_premix/cpremix_velratio0.8000_Z0.1250_Y1.0000.ulf')

settings = {
    'parameters': {
        'Hnorm': {'method': 'linspace', 'values': [0, 1, 5]},
        'velratio': {'method': 'linspace', 'values': [0.1, 1, 10]},
        'Y': {'method': 'linspace', 'values': [0.7, 1.0, 31]},
        'Z': {'method': 'linspace', 'points': 101, 'values': [0, 0.2, 21]}},
    'flut': {
        'c': {'points': 101},
        'pv': {'CO': 1, 'CO2': 1},
        'solver': 'OpenFOAM',
        'varZ': {'method': 'linspace', 'values': [0, 1, 3]},
        'variables': ['rho', 'T']},
    'oxidizer': {'T': {'max': 710, 'min': 310},
                 'X': {'N2': 0.21, 'O2': 0.79}},
    'pressure': 101325,
    'ulf': {'Le1': True,
            'tpatch_end': 0.002,
            'basename': 'cpremix',
            'keys': {'basename': 'RESULT_BASENAME',
                     'chist': 'CHIST',
                     'mechanism': 'MECHANISM',
                     'n_grid': 'REFINE_TARGET',
                     'pressure': 'PRESSURE',
                     'tpatch_end': 'T_PATCH_END',
                     'sl_guess': 'SL_GUESS',
                     'Z': 'Z_VALUE',
                     'T_fix': 'TFIX'},
            'mechanism': 'ch4_smooke.xml',
            'n_grid': 270,
            'par_format': '{:5.4f}',
            'restart': False,
            'T_fix': 20.0,
            'sl_guess': 0.2,
            'solver': '/shared_home/messig/ULF/build_opt/ulf.x',
            'ulf_reference': 'fp_setup.ulf',
            'ulf_reference_bs': 'bs_setup.ulf'},
    'volatiles': {'T': {'max': 600, 'min': 310}, 'Y': {'CH4': 1}}}


@pytest.fixture(scope='module')
def test_path():
    """Setup the path for testing CoalDiffusionFLUT"""
    cwd = os.getcwd()
    path = os.path.abspath(os.path.join(cwd, 'test/test_premix'))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.chdir(path)
    CoalPremixFLUT.copy_files()
    yield path

    # teardown
    os.chdir(cwd)
    shutil.rmtree(path)


@pytest.fixture
def flut(test_path):
    """Init CoalDiffusionFLUT"""
    cflut = CoalPremixFLUT(settings)
    yield cflut

    # teardown
    # remove intermediate files
    [os.remove(f) for f in glob.iglob('inp_*.ulf')]
    [os.remove(f)
     for f in glob.iglob('{}_*.ulf'.format(cflut.basename))]


def test_init(flut):
    runner = UlfRun(flut.ulf_reference, flut.solver)
    assert float(runner[flut.keys['tpatch_end']]) == flut.tpatch_end

    runner_bs = UlfRun(flut.ulf_reference_bs, flut.solver)
    assert float(runner_bs[flut.keys['tpatch_end']]) == flut.tpatch_end


def test_run_set_runner_fp(flut):
    """
    test set_runner for velratio = 1
    """
    parameters = (1, 0.1, 1)
    basename_calc = flut.basename + flut.create_label(parameters) + '_run'
    input_file = 'inp' + flut.create_label(parameters) + '_run.ulf'
    shutil.copy(flut.ulf_reference, input_file)
    runner = flut._run_set_runner(basename_calc, input_file, parameters)

    assert float(runner[flut.keys['Z']]) == parameters[1]
    assert runner[flut.keys['basename']] == basename_calc
    # assert float(runner[flut.keys['T_fix']])
    assert float(runner[flut.keys['sl_guess']]) == flut.sl_guess


def test_run_set_runner_bs(flut):
    '''
    test set_runner for velratio < 1
    '''
    parameters = (0.9, 0.1, 1)

    sL = 0.5
    flut.sL = np.ones((len(flut.Z), len(flut.Y)), dtype=float) * sL
    basename_calc = flut.basename + flut.create_label(parameters) + '_run'
    input_file = 'inp' + flut.create_label(parameters) + '_run.ulf'
    shutil.copy(flut.ulf_reference, input_file)
    runner = flut._run_set_runner(basename_calc, input_file, parameters)

    assert float(runner[flut.keys['Z']]) == parameters[1]
    assert runner[flut.keys['basename']] == basename_calc
    # assert float(runner[flut.keys['T_fix']])
    assert float(runner[flut.keys['sl_guess']]) == sL * parameters[0]


def test_cutflames():
    flame = pyFLUT.Flame1D.read_ulf(ulf_results)
    flamec = CoalPremixFLUT._cut_flame(flame)

    x = flamec['X']
    for index in (0, -1):
        assert x[index] == flame['X'][index]
    assert flame['T'][0] == flamec['T'][0]
    assert flamec['T'][0] < flamec['T'][1]


@pytest.fixture
def flame_exception(flut):
    """
    Generate a quenched flame for managing exception from ULF
    """
    parameters = (1, 0.1, 1)
    basename_calc = flut.basename + flut.create_label(parameters) + "_run"
    ulf_init = basename_calc + 'init.ulf'
    shutil.copy(ulf_results, ulf_init)
    res = flut._exception_ulf_fail(parameters, basename_calc)

    yield res

    os.remove(flut.basename + flut.create_label(parameters) + '.ulf')
    os.remove(ulf_init)


def test_run_exception(flame_exception):
    assert flame_exception.shape[0] == 2
    assert flame_exception['T'][0] == flame_exception['T'][1]
    assert flame_exception['X'][0] != flame_exception['X'][1]
    np.testing.assert_equal(flame_exception['deltah'], 0)
