/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** projectionarea.h
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#ifndef PROJECTIONAREA_H
#define PROJECTIONAREA_H

#include <qstring.h>
#include <qmap.h>

enum ProjType
{
    FRONT,
    BACK,
    LEFT,
    RIGHT,
    TOP,
    BOTTOM
};
class ProjectionArea;

typedef QMap<QString, ProjectionArea> ProjectionAreaMap;

class ProjectionArea
{

public:
    ProjectionArea();
    QString getName();
    ProjType getType();
    int getWidth();
    int getHeight();
    double *getOrigin();
    double getOriginX();
    double getOriginY();
    double getOriginZ();
    double *getRotation();
    double getRotation_h();
    double getRotation_p();
    double getRotation_r();
    void setName(QString n);
    void setType(ProjType t);
    void setWidth(int w);
    void setHeight(int h);
    void setOrigin(double o[3]);
    void setOrigin(double x, double y, double z);
    void setRotation(double r[3]);
    void setRotation(double h, double p, double r);

private:
    QString name;
    ProjType pType;
    int width;
    int height;
    double origin[3];
    double rotation[3];
};
#endif // PROJECTIONAREA_H
