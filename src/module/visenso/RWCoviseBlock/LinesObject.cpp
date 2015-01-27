/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LinesObject.h"
#include <iostream>
#include <do/coDoLines.h>

#if defined(WIN32) || defined(WIN64)
#define COVISE_write _write
#else //WIN32 || WIN64
#define COVISE_write write
#endif //WIN32 || WIN64

LinesObject::LinesObject()
    : OutputObject("Lines")
{
}

LinesObject::LinesObject(const LinesObject &o)
    : OutputObject("Lines")
{
    std::string x = o.type_;
}

LinesObject::LinesObject(const OutputObject &o)
    : OutputObject("Lines")
{
    std::string x = o.type();
}

LinesObject *LinesObject::clone() const
{
    std::cerr << "LinesObject::clone() called  type: " << type_ << std::endl;

    return new LinesObject(*this);
}

LinesObject::~LinesObject()
{
}

bool LinesObject::process(const int &fd)
{
    if (!distrObj_)
        return false;

    if (!distrObj_->isType("LINES"))
    {
        std::cerr << "LinesObject::process() object mismatch: got "
                  << distrObj_->getType() << " but expected LINES"
                  << std::endl;

        return false;
    }

    if (!distrObj_->objectOk())
    {
        std::cerr << "LinesObject::process() object has a shm problem" << std::endl;

        return false;
    }

    coDoLines *lines = (coDoLines *)distrObj_;
    int numLines = lines->getNumLines();
    int numVertices = lines->getNumVertices();
    int numPoints = lines->getNumPoints();

    float *x(NULL), *y(NULL), *z(NULL);
    int *vl(NULL), *ll(NULL);
    lines->getAddresses(&x, &y, &z, &vl, &ll);

    //for (int i = 0; i < numPoints; ++i)
    //   std::cerr << " ## " << x[i] << "  " << y[i] << "  " << z[i] << std::endl;

    COVISE_write(fd, "LIN", 3 * sizeof(char));

    // write lines array
    COVISE_write(fd, &numLines, sizeof(int));
    COVISE_write(fd, ll, numLines * sizeof(int));

    // write index array
    COVISE_write(fd, &numVertices, sizeof(int));
    COVISE_write(fd, vl, numVertices * sizeof(int));

    // write vertex array
    COVISE_write(fd, &numPoints, sizeof(int));
    COVISE_write(fd, x, numPoints * sizeof(float));
    COVISE_write(fd, y, numPoints * sizeof(float));
    COVISE_write(fd, z, numPoints * sizeof(float));

    COVISE_write(fd, "FI", 2 * sizeof(char));

    return true;
}
