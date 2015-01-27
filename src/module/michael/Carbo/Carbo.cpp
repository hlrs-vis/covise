/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file Carbo.cpp

/******************************************************************************\ 
 **                                                              (C)2001 RUS **
 **                                                                          **
 ** Description:  COVISE LineIntegralConvolution application module          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Author: M. Muench                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** xx. ???? 01 v1                                                            **
 ** XXXXXXXXX xx new covise api                                              **
 **                                                                          **
\******************************************************************************/

#include "Carbo.h"

/********************\ 
 *                  *
 * Covise main loop *
 *                  *
\********************/

int main(int argc, char *argv[])
{
    Carbo *application = new Carbo();
    application->start(argc, argv);
    return 0;
}

/******************************\ 
 *                            *
 * Ingredients of Application *
 *                            *
\******************************/

Carbo::Carbo()
{
    // this info appears in the module setup window
    set_module_description("Carbo test data creator");

    //parameters
    resolution = addInt32Param("Resolution", "resolution");
    domainSize = addFloatParam("Domain Size", "size");
    //distortion = addFloatParam("Distortion Parameter", "distortion");

    const int defaultDim = 2;
    resolution->setValue(defaultDim);

    const float defaultSize = 1.0;
    domainSize->setValue(defaultSize);

    //const float defaultDistortion = 0.0;
    //domainSize->setValue(defaultDistortion);

    // the output ports
    polygonOutPort = addOutputPort("polygonOut", "coDoPolygons", "Polygons");
    //gridOutPort = addOutputPort("gridOut","coDoUnstructuredGrid","UGrid");
    vectorOutPort = addOutputPort("vectorOut", "coDoVec3", "Vectors");
}

Carbo::~Carbo()
{
}

void Carbo::quit()
{
}

int Carbo::compute()
{
    int dimension = resolution->getValue();
    float size = domainSize->getValue();
    //float crooked = distortion->getValue();

    coDoVec3 *vectors = NULL;
    coDoPolygons *polygons = NULL;
    //coDoUnstructuredGrid* ugrid;

    doPolygons(&polygons, &vectors, dimension, size);
    /* 
      //begin test
      {
         f2ten m = f2ten(3);
         (m[0]).resize(5, 0.0);
         (m[1]).resize(5, 0.0);
         (m[2]).resize(5, 0.0);

         m[0][0] = 3.0;  m[0][1] = 1.0;   m[0][2] = 6.0;
         m[1][0] = 2.0;  m[1][1] = 1.0;   m[1][2] = 3.0;
         m[2][0] = 1.0;  m[2][1] = 1.0;   m[2][2] = 1.0;

   m = gauss3D(m);

   prF2ten(m);
   }
   //end test
   */
    return SUCCESS;
}

void Carbo::doPolygons(coDoPolygons **polygon,
                       coDoVec3 **vectors,
                       int dimension, float size)
{
    //***************************************************************************

    int num_points = (dimension + 1) * (dimension + 1);
    f2ten coordinates = f2ten(3);
    {
        for (int i = 0; i < 3; i++)
        {
            (coordinates[i]).resize(num_points);
        }
    }

    int num_polygons = 2 * (dimension * dimension); //triangles
    ivec polys = ivec(num_polygons);
    ivec type = ivec(num_polygons);

    int num_corners = 3 * num_polygons; //triangles
    ivec corners = ivec(num_corners);

    float fact = static_cast<float>(dimension);
    fact = size / fact;

    //set coordinates
    {
        //float var = fact*0.3;
        //long rinit = -7;

        for (int i = 0; i < (dimension + 1); i++)
        {
            for (int j = 0; j < (dimension + 1); j++)
            {
                coordinates[0][i * (dimension + 1) + j] = (fact * j);
                //coordinates[0][i*(dimension+1)+j] += var*random2(&rinit);
                coordinates[1][i * (dimension + 1) + j] = (fact * i);
                //coordinates[1][i*(dimension+1)+j] += var*random2(&rinit);
                coordinates[2][i * (dimension + 1) + j] = 0;
            }
        }
    }

    //set polys
    {
        for (int i = 0; i < num_polygons; i++)
        {
            polys[i] = 3 * i;
            type[i] = TYPE_TRIANGLE;
        }
    }

    //set corners
    {
        int i = 0;
        int col = 0;
        while (i < num_corners)
        {
            if (((col + 1) % (dimension + 1)) == 0)
                ++col;
            else
                ;

            corners[i] = col;
            corners[i + 1] = col + 1;
            corners[i + 2] = col + (dimension + 1);

            corners[i + 3] = corners[i + 1] = col + 1;
            corners[i + 4] = (col + 1) + (dimension + 1);
            corners[i + 5] = col + (dimension + 1);

            i += 6;
            ++col;
        }
    }

    {
        int npp = num_points;

        //for Carbo.*
        float ucharge = 1.0;
        fvec ePotential = fvec(npp, 0);
        f2ten eField = f2ten(3);
        //f2ten coordinates = f2ten(3);
        fvec cmax = fvec(3, 0);
        fvec cmin = fvec(3, 0);
        float diameter = 0;

        {
            for (int i = 0; i < 3; i++)
            {
                //(coordinates[i]).resize(npp);
                (eField[i]).resize(npp);
            }
        }
        {
            for (int i = 0; i < npp; i++)
            {
                cmax[0] = FMAX(cmax[0], coordinates[0][i]);
                cmin[0] = FMIN(cmin[0], coordinates[0][i]);
                cmax[1] = FMAX(cmax[1], coordinates[1][i]);
                cmin[1] = FMIN(cmin[1], coordinates[1][i]);
                cmax[2] = FMAX(cmax[2], coordinates[2][i]);
                cmin[2] = FMIN(cmin[2], coordinates[2][i]);
            }
            diameter = abs((cmax - cmin));
            carbonDioxide(ePotential, eField, ucharge, diameter, coordinates);
        }

        (*vectors) = new coDoVec3(vectorOutPort->getObjName(),
                                  npp, &eField[0][0],
                                  &eField[1][0], &eField[2][0]);
        vectorOutPort->setCurrentObject(*vectors);
    }

    (*polygon) = new coDoPolygons(polygonOutPort->getObjName(), num_points,
                                  &coordinates[0][0], &coordinates[1][0],
                                  &coordinates[2][0], num_corners, &corners[0],
                                  num_polygons, &polys[0]);

    polygonOutPort->setCurrentObject(*polygon);
    (*polygon)->addAttribute("vertexOrder", "2");

    /////////////////////////////////////////////////////////////////////
}

void Carbo::carbonDioxide(fvec &ePotential, f2ten &eField, float ucharge,
                          float size, const f2ten &coord)
{
    const int numPoints = (coord[0]).size();

    const float rmin = max((size * 0.002), 0.00001);
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
            //cout << "\n\n" << flush;
            int j;
            for (j = 0; j < 3; j++)
            {
                tmpC1[j] = coord[j][i] - c1[j];
                tmpO1[j] = coord[j][i] - o1[j];
                tmpO2[j] = coord[j][i] - o2[j];
            }
            rC1 = max(rmin, abs(tmpC1));
            rO1 = max(rmin, abs(tmpO1));
            rO2 = max(rmin, abs(tmpO2));
            elpot[i] = ((-2.0) * ucharge / rC1) + (1.0 * ucharge / rO1) + (1.0 * ucharge / rO2);

            r3C1 = pow(rC1, 3);
            r3O1 = pow(rO1, 3);
            r3O2 = pow(rO2, 3);
            for (j = 0; j < 3; j++)
            {
                elfi[j][i] = ((2.0 * ucharge / r3C1) * (tmpC1[j])) + (((-1.0) * ucharge / r3O1)
                                                                      * (tmpO1[j])) + (((-2.0) * ucharge / r3O2) * (tmpO2[j]));

                //cout << "elfi[" << flush << j << flush << "][" << flush;
                //cout << i << flush << "] = " << flush;
                //cout << elfi[j][i] << flush << "   " << flush;
            }
        }
    }

    ePotential = elpot;

    eField = elfi;
}

//returns pseudo random number 0 < ran-num < 1
float Carbo::random2(long *idum)
{
    const int IM1 = 2147483563;
    const int IM2 = 2147483339;
    const float AM = (1.0 / IM1);
    const int IMM1 = (IM1 - 1);
    const int IA1 = 40014;
    const int IA2 = 40692;
    const int IQ1 = 53668;
    const int IQ2 = 52774;
    const int IR1 = 12211;
    const int IR2 = 3791;
    const int NTAB = 32;
    const int NDIV = (1 + (IMM1 / NTAB));
    const float EPS = 1.2e-7;
    const float RNMX = (1.0 - EPS);

    //random number generator of L'Ecuyer with Bays-Durham shuffle
    //and added safeguards. period > 2x10^8; call with idum < 0.

    int j = 0;
    long k = 0;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    float temp = 0.0;

    if (*idum <= 0)
    {
        if (-(*idum) < 1)
        {
            *idum = 1;
        }
        else
        {
            *idum = -(*idum);
        }
        idum2 = (*idum);

        for (j = NTAB + 7; j >= 0; j--) //Load the shuffle table(after 8 warm-ups) ??
        {
            k = (*idum) / IQ1;
            (*idum) = ((IA1 * ((*idum) - (k * IQ1))) - (k * IR1));
            if ((*idum) < 0)
            {
                (*idum) += IM1;
            }
            else
            {
            }
            if (j < NTAB)
            {
                iv[j] = (*idum);
            }
            else
            {
            }
        }

        iy = iv[0];
    }
    else
    {
    }

    k = (*idum) / IQ1; //starting point when not initializing
    (*idum) = ((IA1 * ((*idum) - (k * IQ1))) - (k * IR1));

    if ((*idum) < 0)
    {
        (*idum) += IM1;
    }
    else
    {
    }

    k = idum2 / IQ2;
    idum2 = ((IA2 * (idum2 - (k * IQ2))) - (k * IR2));

    if (idum2 < 0)
    {
        idum2 += IM2;
    }
    else
    {
    }

    j = iy / NDIV;
    iy = iv[j] - idum2;
    iv[j] = (*idum);

    if (iy < 1)
    {
        iy += IMM1;
    }
    else
    {
    }
    if ((temp = AM * iy) > RNMX)
    {
        return RNMX;
    }
    else
    {
        return temp;
    }

    return temp;
}

//matrix(3,5) - col 4 -> row_scaling, col 5 -> col_pivoting !!
f2ten Carbo::gauss3D(f2ten matrix)
{
    /////////////////////////////////////////////////////////////////////

    //check matrix

    if (matrix.size() != 3)
    {
        cerr << "\n... problem in transforming matrix\n" << flush;
        exit(99);
    }
    else if ((matrix[0]).size() != 5 || (matrix[1]).size() != 5 || (matrix[2]).size() != 5)
    {
        cerr << "\n... problem in transforming matrix\n" << flush;
        exit(99);
    }
    else
        ;

    /////////////////////////////////////////////////////////////////////

    //scale matrix

    int i;
    int j;

    fvec max = fvec(3, 0.0);

    for (i = 0; i < 3; i++)
    {
        j = maxIndex(matrix[i]);

        max[i] = fabs(matrix[i][j]);
        matrix[i][3] = max[i];
    }

    for (i = 0; i < 3; i++)
    {
        if (max[i] > 1e-10)
        {
            int j;
            for (j = 0; j < 3; j++)
            {
                matrix[i][j] /= max[i];
            }
        }
        else
            ;
    }

    /////////////////////////////////////////////////////////////////////

    //initialize permutation = matrix[*][4]

    for (i = 0; i < 3; i++)
    {
        matrix[i][4] = i;
    }

    /////////////////////////////////////////////////////////////////////

    //do LU decomposition with col_pivoting & row_scaling

    //first step
    {
        fvec tmp = fvec(3, 0.0);
        tmp[0] = matrix[0][0];
        tmp[1] = matrix[1][0];
        tmp[2] = matrix[2][0];
        int kk = maxIndex(tmp);

        cout << "\nfirst: maxIndex = " << flush << kk << flush;

        //pivoting
        swap(matrix[0], matrix[kk]);

        //transformation
        float lambda = 0.0;

        matrix[1][0] /= matrix[0][0];
        lambda = (-1.0) * matrix[1][0];

        matrix[1][1] += lambda * matrix[0][1];
        matrix[1][2] += lambda * matrix[0][2];

        matrix[2][0] /= matrix[0][0];
        lambda = (-1.0) * matrix[2][0];

        matrix[2][1] += lambda * matrix[0][1];
        matrix[2][2] += lambda * matrix[0][2];
    }

    //second step
    {
        fvec tmp = fvec(2, 0.0);
        tmp[0] = matrix[1][1];
        tmp[1] = matrix[2][1];
        int kk = maxIndex(tmp);
        ++kk;

        //pivoting
        swap(matrix[1], matrix[kk]);

        //transformation
        float lambda = 0.0;

        matrix[2][1] /= matrix[1][1];
        lambda = (-1.0) * matrix[2][1];

        matrix[2][2] += lambda * matrix[1][2];
    }

    return matrix;
}
