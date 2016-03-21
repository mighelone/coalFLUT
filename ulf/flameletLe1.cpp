#include "ulf.h"
using namespace ulf;

extern "C" {
    void updateEnthalpyConfigure( equation * eq )
	{
		EQ_REG_F( "hMean" );
	}
    void updateEnthalpy( equation * eq, field *rhs )
	{
		field &hMean = *EQ_F(0);
		EQ_MIX.update();
	}

    void setEnthalpyProfileConfigure( equation * eq)
	{
		EQ_REG_P( "DHmax" );
		EQ_REG_P( "Zmax" );
		EQ_REG_G("H_o");
		EQ_REG_G("H_f");
	}
    void setEnthalpyProfile( equation * eq, field *rhs )
	{
		int end = rhs->size()-1;
		double &H_o = EQ_G(0)[0];
		double &H_f = EQ_G(1)[0];
		//double H_f = *(hMean->end());
		printf("H_o : %e\n", H_o);
		printf("H_f : %e\n", H_f);
        // get mesh
        mesh &msh = EQ_MSH;

		double DH = EQ_P(0);
		double Zx = EQ_P(1);
		forAllSeq((*rhs), i)
		{
				double z = msh[0][i];
				(*rhs)[i] = H_f * z + (1-z) * H_o;
				if (z < Zx)
				{
						(*rhs)[i] -= DH * z / Zx;
				}
				else
				{
						(*rhs)[i] -= DH * (1.0 - z)/(1.0-Zx);
				}
		}
	}
    void setEnthalpyConfigure( equation * eq)
	{
		EQ_REG_F( "hMean" ); 	//0
		EQ_REG_F( "T" );		//1
		EQ_REG_F( "p" );		//2
		EQ_REG_F( "{Yi}" );     //3 

		EQ_REG_G("H_o");
		EQ_REG_G("H_f");
	}
    void setEnthalpy( equation * eq, field *rhs )
	{
		field &hMean = *EQ_F(0);
		field &T 	 = *EQ_F(1);
		field &p 	 = *EQ_F(2);
		field **Yi   = &EQ_F(3);
		int end = T.size()-1;
		T[0] = EQ_PARSER("T_oxidizer")->getDouble();
		T[end] = EQ_PARSER("T_oxidizer")->getDouble();
		fieldData &Z = EQ_MSH[0];
		FOR(i,1,end)
		{
				double z = Z[i];
				T[i] = T[end] * z + (1-z) * T[0];

		}
		FOR(i,0,EQ_NSPEC)
		{
				(*Yi[i]) = 0.0;

		}
		parser * oxid = EQ_PARSER("oxidizer");
        forAllSeq(*oxid, i)
        {
            char  *name  = (*oxid)[i].getKey();
            double value = (*oxid)[i].getDouble();
            int    iSpec = EQ_MIX.speciesIndex(name);
			if(iSpec<0)
			{
					printf("species %s not in mechanism\n",name);
			}
			(*Yi[iSpec])[0] = value;
		}
		parser * fuel = EQ_PARSER("fuel");
        forAllSeq(*fuel, i)
        {
            char  *name  = (*fuel)[i].getKey();
            double value = (*fuel)[i].getDouble();
            int    iSpec = EQ_MIX.speciesIndex(name);
			if(iSpec<0)
			{
					printf("species %s not in mechanism\n",name);
			}
			(*Yi[iSpec])[end] = value;
		}

		FOR(i,0,EQ_NSPEC)
		{
				field &Y_i = *Yi[i];
				FOR(j,1,end)
				{
						double z = Z[j];
						Y_i[j] = Y_i[end] * z + (1-z) * Y_i[0];

				}

		}

		Cantera::IdealGasMix *gas = (Cantera::IdealGasMix *) EQ_MIX.getCanteraObject();
		FOR(i,1,end)
		{
          double Y[EQ_NSPEC];
          for(int j = 0; j < EQ_NSPEC; j++)
          {
              Y[j] = (*Yi[j])[i];
          }
          gas->setState_TPY(T[i], p[i], Y);
		  gas->equilibrate("HP");
		  T[i] = gas->temperature();
  		  gas->getMassFractions(Y);
          for(int j = 0; j < EQ_NSPEC; j++)
          {
              (*Yi[j])[i] = Y[j];
          }

		}
        EQ_MIX.update();
		EQ_G(0)[0] = hMean[0];
		EQ_G(1)[0] = hMean[end];
		equation calcEnthalpy(EQ_PRB.getParser()->search(".fields.hMean"),&EQ_PRB,1,1,&hMean);
		calcEnthalpy.evaluate(&hMean);
		EQ_MIX.setUpdateEnthalpy(1);
		EQ_MIX.update();
	}

    ////////////////////////////////////////////////////////////
    //EQUATION T_flameletLe1 
    //EQUATION T_flameletLe2 
    // flamelet Le = 1 - standard temperature equation
    //
    ////////////////////////////////////////////////////////////

    void T_flameletLe1DHConfigure( equation * eq )
    {
        EQ_CREATE_F( "scratch"  );

		EQ_REG_F( "T" );        // 0
		EQ_REG_F( "tdot" );     // 1
		EQ_REG_F( "chi" );      // 2
		EQ_REG_F( "cpMean" );   // 3
		EQ_REG_F( "scratch" );  // 4
		EQ_REG_F( "{Yi}" );     // 5          ... 4+nSpecies-1
		EQ_REG_F( "cp_{Yi}" );  // 5+nSpecies ... 4+2*nSpecies-1
		
		EQ_REG_P( "DHmax" );
    }

    void T_flameletLe1DH( equation * eq, field *rhs )
    {
        // get nSpecies for problem dimensions
        int nSpecies = EQ_NSPEC;

        // get fields
        field &dTdt = *rhs;
        field &T = *EQ_F(0);
        field &tdot = *EQ_F(1);
        field &chi = *EQ_F(2);
        field &cpMean = *EQ_F(3);
        field &scratch = *EQ_F(4);
        field **Yi = &EQ_F(5);
        field **cpi = &EQ_F(5 + nSpecies);

        field sumTerm3 = field(T.size(), 0.0);
        for(int i = 0; i < nSpecies; i++)
        {
            field dYidZ = Yi[i]->derivative(0, FIRST, cpi[i]);
            sumTerm3   += *(cpi[i]) * dYidZ;
        }

        // compose equation
        field Term2 = 0.5 * chi * T.derivative(0, SECOND);
        scratch     = 0.5 * chi / cpMean * (sumTerm3 + cpMean.derivative(0, FIRST));
        field Term3 = scratch * T.derivative(0, FIRST, &scratch);
        field Term4 = tdot;
		
        // get mesh
        mesh &msh = EQ_MSH;

        field Term5 = field(T.size(), 0.0);
		double Zx = 0.1;
		double DH = EQ_P(0);
		forAllSeq(Term5, i)
		{
				if (msh[0][i] < Zx)
				{
						Term5[i] = DH*msh[0][i]/Zx;
				}
				else
				{
						Term5[i] = DH*(1-msh[0][i])/(1.0-Zx);
				}
		}
		

        dTdt = Term2 + Term3 + Term4 -Term5/cpMean;
    }

}
