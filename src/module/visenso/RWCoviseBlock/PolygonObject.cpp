/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PolygonObject.h"
#include <iostream>

#if defined(WIN32) || defined(WIN64)
#define COVISE_write _write
#else //WIN32 || WIN64
#include <unistd.h>
#define COVISE_write write
#endif //WIN32 || WIN64

#include <do/coDoPolygons.h>

PolygonObject::PolygonObject()
    : OutputObject("POLYGN")
{
}

PolygonObject::PolygonObject(const PolygonObject &o)
    : OutputObject(o)
{
}

PolygonObject::PolygonObject(const OutputObject &o)
    : OutputObject("POLYGN")
{
    std::string x = o.type();
}

bool PolygonObject::process(const int &fd)
{
    if (!distrObj_)
        return false;

    if (!distrObj_->isType("POLYGN"))
    {
        std::cerr << "PolygonObject::process() object mismatch POLYGN expecxted" << std::endl;
        std::cerr << "PolygonObject::process() got " << distrObj_->getType() << std::endl;

        return false;
    }

    if (!distrObj_->objectOk())
    {
        std::cerr << "PolygonObject::process() object has a shm problem" << std::endl;

        return false;
    }

    coDoPolygons *poly = (coDoPolygons *)distrObj_;
    int numPoly = poly->getNumPolygons();
    int numPts = poly->getNumPoints();
    int numVert = poly->getNumVertices();
    float *x(NULL), *y(NULL), *z(NULL);
    int *cl(NULL), *pl(NULL);
    poly->getAddresses(&x, &y, &z, &cl, &pl);
    int i = 0;
    int a = fd;
    ++a;

    //for (i=0; i<numPts; ++i)
    //   std::cerr << " ## " << x[i] << "  " << y[i] << "  " << z[i] << std::endl;

    // check for quads .. nad write idxArr
    int nIdxArr = 3 * numPoly;
    int *idxArr = new int[nIdxArr];
    bool allTri = true;
    int idx = 0;

    for (i = 0; i < numPoly - 1; ++i)
    {
        allTri = allTri && ((pl[i + 1] - pl[i]) == 3);
        if (!allTri)
        {
            std::cerr << " PRIMITIVES OTHER THEN TRIANGLES FOUND" << std::endl;

            return false;
        }

        idxArr[idx] = cl[pl[i]];
        idx++;
        idxArr[idx] = cl[pl[i] + 1];
        idx++;
        idxArr[idx] = cl[pl[i] + 2];
        idx++;
    }

    allTri = allTri && ((numVert - pl[numPoly - 1]) == 3);

    if (!allTri)
    {
        std::cerr << " PRIMITIVES OTHER THEN TRIANGLES FOUND" << std::endl;

        return false;
    }

    idxArr[idx] = cl[pl[numPoly - 1]];
    idx++;
    idxArr[idx] = cl[pl[numPoly - 1] + 1];
    idx++;
    idxArr[idx] = cl[pl[numPoly - 1] + 2];
    idx++;

    COVISE_write(fd, "TRI", 3 * sizeof(char));
    COVISE_write(fd, &numPts, sizeof(int));
    COVISE_write(fd, x, numPts * sizeof(float));
    COVISE_write(fd, y, numPts * sizeof(float));
    COVISE_write(fd, z, numPts * sizeof(float));
    COVISE_write(fd, &nIdxArr, sizeof(int));
    COVISE_write(fd, idxArr, nIdxArr * sizeof(int));
    COVISE_write(fd, "FI", 2 * sizeof(char));

    delete[] idxArr;

    return true;
}

PolygonObject *PolygonObject::clone() const
{
    std::cerr << "PolygonObject::clone() called  type: " << type_ << std::endl;

    return new PolygonObject(*this);
}

PolygonObject::~PolygonObject()
{
}
