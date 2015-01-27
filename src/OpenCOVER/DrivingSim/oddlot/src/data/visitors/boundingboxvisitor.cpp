/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/19/2010
**
**************************************************************************/

#include "boundingboxvisitor.hpp"
#ifdef WIN32
#define isnan _isnan
#endif

#include <math.h>
#include <float.h>

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/track/trackcomposite.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"

#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/scenerymap.hpp"

// Utils //
//
#include "math.h"

// NOTE:this is just an approximation!

BoundingBoxVisitor::BoundingBoxVisitor()
{
}

void
BoundingBoxVisitor::visit(RoadSystem *roadSystem)
{
    roadSystem->acceptForRoads(this);
}

void
BoundingBoxVisitor::visit(RSystemElementRoad *road)
{
    road->acceptForTracks(this);
}

void
BoundingBoxVisitor::visit(TrackComposite *acceptor)
{
    acceptor->acceptForChildNodes(this);
}

void
BoundingBoxVisitor::visit(TrackElement *acceptor)
{
    QPointF pointA = acceptor->getGlobalPoint(acceptor->getSStart());
    QPointF pointB = acceptor->getGlobalPoint(acceptor->getSEnd());
    //	qDebug() << pointA << " " << pointB;
    double width = fabs(pointB.x() - pointA.x());
    double height = fabs(pointB.y() - pointA.y());
    double x = (pointA.x() < pointB.x()) ? pointA.x() : pointB.x();
    double y = (pointA.y() < pointB.y()) ? pointA.y() : pointB.y();
    //	qDebug() << QRectF(x, y, width, height) << "\n";
    if (!(isnan(x) || isnan(y) || isnan(width) || isnan(height)))
        boundingBox_ = boundingBox_.united(QRectF(x, y, width, height));
}

//void
//	BoundingBoxVisitor
//	::visit(TrackElementLine * acceptor)
//{

//}

//void
//	BoundingBoxVisitor
//	::visit(TrackElementArc * acceptor)
//{

//}

//void
//	BoundingBoxVisitor
//	::visit(TrackElementSpiral * acceptor)
//{

//}

//void
//	BoundingBoxVisitor
//	::visit(TrackElementPoly3 * acceptor)
//{

//}

void
BoundingBoxVisitor::visit(ScenerySystem *acceptor)
{
    acceptor->acceptForChildNodes(this);
}

void
BoundingBoxVisitor::visit(SceneryMap *acceptor)
{
    boundingBox_ = boundingBox_.united(QRectF(acceptor->getX(), acceptor->getY(), acceptor->getWidth(), acceptor->getHeight()));
}
