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

    const int defaultDim = 2;
    resolution->setValue(defaultDim);

    const float defaultSize = 1.0;
    domainSize->setValue(defaultSize);

    // the output ports
    polygonOutPort = addOutputPort("polygonOut", "coDoPolygons", "Polygons");
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

    coDoVec3 *vectors = NULL;
    coDoPolygons *polygons = NULL;

    doPolygons(&polygons, &vectors, dimension, size);

    return SUCCESS;
}

void Carbo::doPolygons(coDoPolygons **polygon,
                       DO_Unstructured_V3d_Data **vectors,
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

    int num_corners = 3 * num_polygons; //triangles
    ivec corners = ivec(num_corners);

    float fact = static_cast<float>(dimension);
    fact = size / fact;

    //set coordinates
    {
        float var = fact * 0.1;
        long rinit = -7;

        for (int i = 0; i < (dimension + 1); i++)
        {
            for (int j = 0; j < (dimension + 1); j++)
            {
                coordinates[0][i * (dimension + 1) + j] = (fact * j);
                coordinates[0][i * (dimension + 1) + j] += var * random2(&rinit);
                coordinates[1][i * (dimension + 1) + j] = (fact * i);
                coordinates[1][i * (dimension + 1) + j] += var * random2(&rinit);
                coordinates[2][i * (dimension + 1) + j] = 0;
            }
        }
    }

    //set polys
    {
        for (int i = 0; i < num_polygons; i++)
        {
            polys[i] = 3 * i;
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
        f2ten coordinates = f2ten(3);
        fvec cmax = fvec(3, 0);
        fvec cmin = fvec(3, 0);
        float diameter = 0;

        {
            for (int i = 0; i < 3; i++)
            {
                (coordinates[i]).resize(npp);
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
    }

    (*polygon) = new coDoPolygons(polygonOutPort->getObjName(), num_points,
                                  &coordinates[0][0], &coordinates[1][0],
                                  &coordinates[2][0], num_corners, &corners[0],
                                  num_polygons, &polys[0]);

    polygonOutPort->setCurrentObject(*polygon);
    (*polygon)->addAttribute("vertexOrder", "2");
}
