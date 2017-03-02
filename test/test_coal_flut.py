import coalFLUT
import pytest
import numpy as np
import mock
import pyFLUT

# input_yml = 'examples/Le1/input.yml'
# input_yml = 'test/input.yml'
input_yml = 'input.yml'


@pytest.fixture
def flut():
    return coalFLUT.CoalFLUT(input_yml)


def mixing(p, v, alphac, Y):
    alphac_one = 1 + alphac
    return ((alphac_one * (1 - Y) * p + Y * v) /
            (alphac_one * (1 - Y) + Y))


def test_init(flut):
    assert 'volatiles' in flut.streams


def test_chargases(flut):
    # flut.set_chargas()
    assert hasattr(flut, 'chargases')
    assert (flut.chargases['T'] == flut.volatiles['T']).all()

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


def test_mixfuel_Y0(flut):
    mix = flut.mix_streams(Y=0)
    assert np.allclose(mix['T'], flut.chargases['T'])
    assert np.allclose(mix['H'], flut.chargases['H'])
    for sp in flut.chargases['Y']:
        assert mix['Y'][sp] == flut.chargases['Y'][sp]


def test_mixfuel_Y1(flut):
    mix = flut.mix_streams(Y=1)
    assert np.allclose(mix['T'], flut.volatiles['T'])
    assert np.allclose(mix['H'], flut.volatiles['H'])
    for sp in flut.volatiles['Y']:
        assert mix['Y'][sp] == flut.volatiles['Y'][sp]


def test_mixfuel_Y(flut):
    Y = 0.5
    mix = flut.mix_streams(Y=Y)
    alphac = flut.alphac
    chargases = flut.chargases
    volatiles = flut.volatiles

    assert np.allclose(mix['H'],
                       mixing(chargases['H'],
                              volatiles['H'], alphac, Y))

    for sp in ('CH4', 'CO'):
        assert mix['Y'][sp] == mixing(
            chargases['Y'].get(sp, 0),
            volatiles['Y'].get(sp, 0),
            alphac,
            Y)


def run(fuel, oxidizer, parameters, par_format, ulf_reference, solver,
        species, key_names, basename='res', rerun=True):
    '''
    Mockup run function, returns a zero data structure
    '''
    output_dict = ['Z', 'T', 'rho', 'CO', 'CH4', 'O2', 'N2']
    ngrid = 11
    data = np.zeros((ngrid, len(output_dict)))
    data[:, 0] = np.linspace(0, 1, ngrid)
    n_sp = 3
    for i, sp in enumerate(output_dict[n_sp:], n_sp):
        data[:, i] = np.linspace(oxidizer['Y'].get(sp, 0),
                                 fuel['Y'].get(sp, 0), ngrid)

    return pyFLUT.Flame1D(output_dict=output_dict, data=data,
                          variables=parameters)


@mock.patch('coalFLUT.flut.pyFLUT.ulf.dflut.run_sldf', side_effect=run)
def test_run(mocked_run_sldf, flut):
    # p = {'Y': 0.1, 'Hnorm': 0.5, 'chist': 0.1}
    # res = coalFLUT.pyFLUT.ulf.dflut.run_sldf(fuel=None,
    #                                         oxidizer=None,
    #                                         parameters=p,
    #                                         par_format=None,
    #                                         ulf_reference=None,
    #                                         solver=None,
    #                                         species=None,
    #                                         key_names=None,
    #                                         basename=None)
    # assert res.variables == p
    # assert np.all(res['T'] == 0)
    flut.run()
    assert 'Z' in flut
    for v in ['Z', 'Hnorm', 'Y', 'chist']:
        assert v in flut.input_variables
    assert flut.ndim == 5
    # check if T == 0
    assert (flut['T'] == 0).all()

    alphac = flut.alphac
    vol = flut.volatiles['Y']
    ch = flut.chargases['Y']
    ox = flut.oxidizer['Y']
    species = ['CO', 'CH4', 'O2', 'N2']

    # this method works only if Y is defined between 0 and 1
    Y = flut.input_variable_values('Y')[1]
    for sp in species:
        vol, ch, ox = (s.get(sp, 0) for s in (
            flut.volatiles['Y'], flut.chargases['Y'],
            flut.oxidizer['Y']))
        assert flut.extract_values(sp, Z=1, Y=1) == vol
        assert flut.extract_values(sp, Z=1, Y=0) == ch
        assert flut.extract_values(sp, Z=0, Y=0) == ox
        assert flut.extract_values(
            sp, Z=1, Y=Y) == mixing(ch, vol, alphac, Y)
