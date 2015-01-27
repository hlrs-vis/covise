/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** channel.h
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#ifndef CHANNEL_H
#define CHANNEL_H

#include <qstring.h>
#include <qmap.h>
#include <qptrlist.h>
#include "projectionarea.h"

class Channel
{
public:
    Channel();
    void setIndex(int i);
    void setName(QString s);
    void setLeft(double i);
    void setRight(double i);
    void setBottom(double i);
    void setTop(double i);
    void setProjectionArea(ProjectionArea *p);
    //void setProjector(QString s);

    int getIndex();
    QString getName();
    double getLeft();
    double getRight();
    double getBottom();
    double getTop();
    ProjectionArea *getProjectionArea();
    //QString getProjector();

private:
    int index;
    QString name;
    double left;
    double right;
    double bottom;
    double top;
    ProjectionArea *pArea;
    //QString projector;
};
#endif // CHANNEL_H
