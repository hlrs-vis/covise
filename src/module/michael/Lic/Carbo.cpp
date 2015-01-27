/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file Carbo.cpp

#include "Carbo.h"

void carbonDioxide(fvec &ePotential, f2ten &eField, float ucharge,
                   float size, const f2ten &coord)
{
    const int numPoints = (coord[0]).size();

    const float rmin = fmax((size * 0.002), 0.00001);
    fvec c1 = fvec(3);
    c1[0] = size / 2;
    c1[1] = size / 2;
    c1[2] = 0;
    fvec o1 = fvec(3);
    o1[0] = c1[0] - size / 8;
    o1[1] = c1[1];
    o1[2] = 0;
    fvec o2 = fvec(3);
    o2[0] = c1[0] + size / 8;
    o2[1] = c1[1];
    o2[2] = 0;

    fvec elpot = fvec(numPoints);
    f2ten elfi = f2ten(3);
    {
        for (int j = 0; j < 3; j++)
        {
            elfi[j].resize(numPoints);
        }
    }

    {
        float rC1 = 0.0;
        float rO1 = 0.0;
        float rO2 = 0.0;
        float r3C1 = 0.0;
        float r3O1 = 0.0;
        float r3O2 = 0.0;
        fvec tmpC1 = fvec(3);
        fvec tmpO1 = fvec(3);
        fvec tmpO2 = fvec(3);
        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                tmpC1[j] = coord[j][i] - c1[j];
                tmpO1[j] = coord[j][i] - o1[j];
                tmpO2[j] = coord[j][i] - o2[j];
            }
            rC1 = fmax(rmin, abs(tmpC1));
            rO1 = fmax(rmin, abs(tmpO1));
            rO2 = fmax(rmin, abs(tmpO2));
            elpot[i] = ((-2.0) * ucharge / rC1) + (1.0 * ucharge / rO1) + (1.0 * ucharge / rO2); //electric potential

            r3C1 = pow(rC1, 3);
            r3O1 = pow(rO1, 3);
            r3O2 = pow(rO2, 3);
            for (int j = 0; j < 3; j++)
            {
                elfi[j][i] = ((2.0 * ucharge / r3C1) * (tmpC1[j])) + (((-1.0) * ucharge / r3O1)
                                                                      * (tmpO1[j])) + (((-2.0) * ucharge / r3O2) * (tmpO2[j]));
            }
        }
    }

    ePotential = elpot;

    eField = elfi;
}
