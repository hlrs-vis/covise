/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** channel.cpp
 ** 2004-01-29, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include "channel.h"

Channel::Channel()
{
    index = 0;
    name = "";
    left = 0.0;
    right = 1.0;
    bottom = 0.0;
    top = 1.0;
    pArea = new ProjectionArea();
    pArea->setName("NONE");
    //projector = "LEFT";
}

void Channel::setIndex(int i)
{
    index = i;
}

void Channel::setName(QString s)
{
    name = s;
}

void Channel::setLeft(double i)
{
    left = i;
}

void Channel::setRight(double i)
{
    right = i;
}

void Channel::setBottom(double i)
{
    bottom = i;
}

void Channel::setTop(double i)
{
    top = i;
}

void Channel::setProjectionArea(ProjectionArea *p)
{
    pArea = p;
}

/*void Channel::setProjector(QString s)
{
  projector = s;
}*/

int Channel::getIndex()
{
    return index;
}

QString Channel::getName()
{
    return name;
}

double Channel::getLeft()
{
    return left;
}

double Channel::getRight()
{
    return right;
}

double Channel::getBottom()
{
    return bottom;
}

double Channel::getTop()
{
    return top;
}

ProjectionArea *Channel::getProjectionArea()
{
    return pArea;
}

/*QString Channel::getProjector()
{
  return projector;
}*/
