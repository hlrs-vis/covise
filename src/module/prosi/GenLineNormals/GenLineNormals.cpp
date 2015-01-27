/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GenLineNormals.h"

#include <iostream>
#include <sstream>
#include <list>

#include <util/coMatrix.h>
#include <util/coVector.h>

using namespace std;

#define GENLINENORMALS_XDEFAULT 0.0f
#define GENLINENORMALS_YDEFAULT 0.0f
#define GENLINENORMALS_ZDEFAULT 1.0f

GenLineNormals::GenLineNormals()
    : coModule("Estimate normals from lines")
{

    set_module_description("Tries to generate normals from lines using their curvature");

    linesInPort = addInputPort("lines", "coDoLines", "Lines");
    normalsOutPort = addOutputPort("normals", "DO_Unstructured_V3D_Normals", "Normals");
}

GenLineNormals::~GenLineNormals()
{
}

int GenLineNormals::compute()
{

    coDoVec3 *normals = 0;
    coDoLines *lines = 0;

    string inType = linesInPort->getCurrentObject()->getType();

    if (inType == "LINES")
    {

        lines = static_cast<coDoLines *>(linesInPort->getCurrentObject());

        int noOfPoints = lines->getNumPoints();
        int noOfLines = lines->getNumLines();

        float *nx;
        float *ny;
        float *nz;

        int line = 0;

        int *cornerList;
        int *lineList;
        float *lx;
        float *ly;
        float *lz;

        lines->getAddresses(&lx, &ly, &lz, &cornerList, &lineList);

        normals = new coDoVec3(normalsOutPort->getObjName(), noOfPoints);
        normals->getAddresses(&nx, &ny, &nz);

        nx[0] = GENLINENORMALS_XDEFAULT;
        ny[0] = GENLINENORMALS_YDEFAULT;
        nz[0] = GENLINENORMALS_ZDEFAULT;

        cerr << "Line " << line << endl;

        coVector pred;
        coVector curr(lx[cornerList[0]], ly[cornerList[0]], lz[cornerList[0]]);
        coVector succ(lx[cornerList[1]], ly[cornerList[1]], lz[cornerList[1]]);

        coVector vec1;
        coVector vec2 = succ - curr;

        vec2.normalize();
        coVector norm(nx[0], ny[0], nz[0]);

        for (int normal = 1; normal < noOfPoints - 1; ++normal)
        {

            // Start new line, set the predecessor normal to default...
            if ((line < noOfLines) && (normal == lineList[line + 1]))
            {
                nx[normal - 1] = GENLINENORMALS_XDEFAULT;
                ny[normal - 1] = GENLINENORMALS_YDEFAULT;
                nz[normal - 1] = GENLINENORMALS_ZDEFAULT;
                nx[normal] = GENLINENORMALS_XDEFAULT;
                ny[normal] = GENLINENORMALS_YDEFAULT;
                nz[normal] = GENLINENORMALS_ZDEFAULT;
                ++line;
                cerr << "Line " << line << endl;

                continue;
            }

            pred = curr;
            curr = succ;
            succ = coVector(lx[cornerList[normal + 1]], ly[cornerList[normal + 1]], lz[cornerList[normal + 1]]);

            vec1 = vec2;
            vec2 = succ - curr;

            vec2.normalize();

            coVector tmp1 = vec1.cross(normal);
            coVector tmp2 = vec1.cross(vec2);

            double theta = asin(tmp2.length());

            tmp1.normalize();
            tmp2.normalize();

            if (fabs(theta) > 0.00000001)
            {

                if (vec1 * vec2 < 0.0)
                {
                    theta = copysign(M_PI, theta) - theta;
                }

                coVector tmp3 = vec1 + vec2;
                tmp3.normalize();

                double f1 = tmp2 * norm;
                double f2 = 1.0 - f1 * f1;

                if (f2 > 0.0)
                {
                    f2 = sqrt(1.0 - f1 * f1);
                }
                else
                {
                    f2 = 0.0;
                }

                tmp1 = tmp3.cross(tmp2);
                tmp3 = vec1.cross(tmp2);

                if ((norm * tmp3) * (tmp1 * tmp3) < 0.0)
                {
                    f2 = -f2;
                }

                norm = (tmp2 * f1) + (tmp1 * f2);
            }

            if (norm.isZero())
            {

                ostringstream info;

                info << "Normal " << normal << " " << norm << " is zero" << endl;
                sendWarning(info.str().c_str());
            }

            nx[normal] = norm[0];
            ny[normal] = norm[1];
            nz[normal] = norm[2];
        }

        nx[noOfPoints - 1] = GENLINENORMALS_XDEFAULT;
        ny[noOfPoints - 1] = GENLINENORMALS_YDEFAULT;
        nz[noOfPoints - 1] = GENLINENORMALS_ZDEFAULT;
    }
    else
    {

        ostringstream info;

        info << "Type \"" << inType << "\" not supported.";
        sendError(info.str().c_str());

        return FAIL;
    }

    normalsOutPort->setCurrentObject(normals);

    return SUCCESS;
}

int main(int argc, char *argv[])
{

    GenLineNormals app;
    app.start(argc, argv);
    return 0;
}
