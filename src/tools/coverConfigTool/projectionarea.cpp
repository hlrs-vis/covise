/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** projectionarea.cpp
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include "projectionarea.h"

ProjectionArea::ProjectionArea()
{
    name = "";
    pType = FRONT;
    width = 0;
    height = 0;
    origin[0] = 0.0;
    origin[1] = 0.0;
    origin[2] = 0.0;
    rotation[0] = 0.0;
    rotation[1] = 0.0;
    rotation[2] = 0.0;
}

QString ProjectionArea::getName()
{
    return name;
}

ProjType ProjectionArea::getType()
{
    return pType;
}

int ProjectionArea::getWidth()
{
    return width;
}

int ProjectionArea::getHeight()
{
    return height;
}

double *ProjectionArea::getOrigin()
{
    return origin;
}

double ProjectionArea::getOriginX()
{
    return origin[0];
}

double ProjectionArea::getOriginY()
{
    return origin[1];
}

double ProjectionArea::getOriginZ()
{
    return origin[2];
}

double *ProjectionArea::getRotation()
{
    return rotation;
}

double ProjectionArea::getRotation_h()
{
    return rotation[0];
}

double ProjectionArea::getRotation_p()
{
    return rotation[1];
}

double ProjectionArea::getRotation_r()
{
    return rotation[2];
}

void ProjectionArea::setName(QString n)
{
    name = n;
}

void ProjectionArea::setType(ProjType t)
{
    pType = t;
}

void ProjectionArea::setWidth(int w)
{
    width = w;
}

void ProjectionArea::setHeight(int h)
{
    height = h;
}

void ProjectionArea::setOrigin(double o[3])
{
    origin[0] = o[0];
    origin[1] = o[1];
    origin[2] = o[2];
}

void ProjectionArea::setOrigin(double x, double y, double z)
{
    origin[0] = x;
    origin[1] = y;
    origin[2] = z;
}

void ProjectionArea::setRotation(double r[3])
{
    rotation[0] = r[0];
    rotation[1] = r[1];
    rotation[2] = r[2];
}

void ProjectionArea::setRotation(double h, double p, double r)
{
    rotation[0] = h;
    rotation[1] = p;
    rotation[2] = r;
}
