"""
Test coalFLUT class
"""
import pytest
import coalFLUT
import yaml
import numpy as np
import os
import pyFLUT.ulf as ulf

yaml_file = "files/input.yml"

def init_class():
    return coalFLUT.coalFLUT(input_yaml=yaml_file)

def read_yaml():
    with open(yaml_file, "r") as f: inp = yaml.load(f)
    return inp

def test_init():
    #res = init_class()
    inp = read_yaml()
    # set Y to list
    yaml_test = 'test.yml'
    # set Y as list
    inp['mixture_fraction']['Y']['method'] = 'list'
    inp['mixture_fraction']['Y']['values'] = np.linspace(0, 1, 21).tolist()
    with open(yaml_test, "w") as f: yaml.dump(inp, f)
    res = coalFLUT.coalFLUT(input_yaml=yaml_test)
    assert (inp['mixture_fraction']['Y']['values'] == res.Y).all()
    # set Y as linspace
    inp['mixture_fraction']['Y']['method'] = 'linspace'
    inp['mixture_fraction']['Y']['values'] = [0, 1, 21]
    with open(yaml_test, "w") as f: yaml.dump(inp, f)
    res = coalFLUT.coalFLUT(input_yaml=yaml_test)
    assert (np.linspace(*inp['mixture_fraction']['Y']['values']) == res.Y).all()
    os.remove(yaml_test)

def test_chargas():
    res = init_class()
    assert sum(res.chargas.values()) == 1

def test_run():
    inp = read_yaml()
    fuel = {
        'T': 600,
        'H':1e8,
        'Y': {
            'CH4': 0.5,
            'CH2': 0.5,
            'CO': 0,
            'CO2': 0
        }
    }
    ox = {
        'T': 400,
        'Y': {
            'O2': 0.23,
            'N2': 0.77
        }
    }

    runner = ulf.UlfRun(inp['ulf']['basename']+".ulf", inp['ulf']['solver'])
    runner.set('MECHANISM', inp['mechanism'])
    Y = 0.2
    chist = 10
    Tf = fuel['T']
    Hf = fuel['H']
    res = coalFLUT.runUlf(inp['ulf'], Y, chist, fuel, ox)
    assert isinstance(res, ulf.UlfData)
    assert res.variables['Hf'] == Hf
    assert res.variables['chist'] == chist
    assert res.variables['Y'] == Y



def test_read_dictionary():
    x = [0, 1, 2]
    assert (coalFLUT.read_dict_list(method='list', values=x) == x).all()
    x = [0, 1, 101]
    assert (coalFLUT.read_dict_list(method='linspace', values=x) == np.linspace(*x)).all()
    x = [0, 1, 0.1]
    assert (coalFLUT.read_dict_list(method='arange', values=x) == np.arange(*x)).all()
    x = {'method': 'linspace', 'values': [0, 1, 11]}
    assert (coalFLUT.read_dict_list(**x) == np.linspace(*x['values'])).all()
    x = {'values': [0, 1, 11], 'method': 'linspace'}
    assert (coalFLUT.read_dict_list(**x) == np.linspace(*x['values'])).all()

def test_normalize():
    X = {'CO':1, 'CO2':0.5}
    assert sum(coalFLUT.normalize(X).values()) == 1

def test_mix_fuels():
    res = init_class()
    assert res.mix_fuels(1)['CH4'] == res.volatiles['Y']['CH4']
    assert res.mix_fuels(0)['CO'] == res.chargas['Y']['CO']

def test_hf():
    res = init_class()