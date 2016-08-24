import coalFLUT
import pytest
import numpy as np

input_yml = 'input.yml'


@pytest.fixture
def flut():
    return coalFLUT.CoalFLUTnew(input_yml)


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
    assert alpha == flut.alphac

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
    alphac_one = 1 + alphac

    def mixing(p, v):
        return ((alphac_one * (1 - Y) * p + Y * v) /
                (alphac_one * (1 - Y) + Y))

    chargases = flut.chargases
    volatiles = flut.volatiles

    assert np.allclose(mix['H'], mixing(chargases['H'], volatiles['H']))

    for sp in ('CH4', 'CO'):
        assert mix['Y'][sp] == mixing(
            chargases['Y'].get(sp, 0), volatiles['Y'].get(sp, 0))
