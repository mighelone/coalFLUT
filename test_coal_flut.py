import coalFLUT
import pytest

input_yml = 'input.yml'


@pytest.fixture
def flut():
    return coalFLUT.CoalFLUTnew(input_yml)


def test_init(flut):
    assert 'volatiles' in flut.streams


def test_chargases(flut):
    flut.set_chargas()
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
