import coalFLUT.equilibrium

if __name__ == '__main__':
    flut = coalFLUT.equilibrium.CoalFLUTEq('input_eq.yml')
    flut.run_scoop()
    print(flut)
