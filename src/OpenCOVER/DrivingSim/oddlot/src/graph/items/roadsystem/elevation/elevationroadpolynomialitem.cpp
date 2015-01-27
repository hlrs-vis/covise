/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   24.06.2010
**
**************************************************************************/

#include "elevationroadpolynomialitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/profilegraph.hpp"

#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "elevationsectionpolynomialitem.hpp"
#include "elevationmovehandle.hpp"

// Editor //
//
#include "src/graph/editors/elevationeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsScene>
#include <QTransform>

ElevationRoadPolynomialItem::ElevationRoadPolynomialItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , elevationEditor_(NULL)
    , moveHandles_(NULL)
{
    init();
}

ElevationRoadPolynomialItem::~ElevationRoadPolynomialItem()
{
    // Do not use the garbage in destructors (disposal is already running)
}

void
ElevationRoadPolynomialItem::init()
{
    // ElevationEditor //
    //
    elevationEditor_ = dynamic_cast<ElevationEditor *>(getProfileGraph()->getProjectWidget()->getProjectEditor());
    if (!elevationEditor_)
    {
        qDebug("Warning 1006241105! ElevationRoadPolynomialItem not created by an ElevationEditor");
    }

    foreach (ElevationSection *section, getRoad()->getElevationSections())
    {
        // SectionItem //
        //
        new ElevationSectionPolynomialItem(this, section);
    }

    // MoveHandles //
    //
    moveHandles_ = new QGraphicsPathItem(this);
    rebuildMoveHandles();
}

QRectF
ElevationRoadPolynomialItem::boundingRect() const
{
    QRectF boundingBox;
    foreach (QGraphicsItem *childItem, childItems())
    {
        QGraphicsPathItem *pathItem = static_cast<QGraphicsPathItem *>(childItem);
        boundingBox = boundingBox.united(pathItem->path().boundingRect());
        //		boundingBox = boundingBox.united(childItem->boundingRect());
    }
    return boundingBox;
}

QRectF
ElevationRoadPolynomialItem::translate(qreal x, qreal y)
{
    QTransform qTransformMatrix;
    qTransformMatrix.translate(x, y);

    foreach (QGraphicsItem *childItem, childItems())
    {
        QGraphicsPathItem *pathItem = static_cast<QGraphicsPathItem *>(childItem);
        pathItem->setTransform(qTransformMatrix);
    }

    QRectF boundingBox = qTransformMatrix.mapRect(boundingRect());

    return boundingBox;
}

void
ElevationRoadPolynomialItem::rebuildMoveHandles()
{
    deleteMoveHandles();

    // Have fun understanding this //
    //
    ElevationSection *previousSection = NULL;
    ElevationMoveHandle *previousHandle = NULL;
    foreach (ElevationSection *section, getRoad()->getElevationSections())
    {
        ElevationMoveHandle *handle = new ElevationMoveHandle(elevationEditor_, moveHandles_);
        handle->registerHighSlot(section);
        if (previousSection)
        {
            handle->registerLowSlot(previousSection);
        }

        // Handle Degrees of freedom //
        //
        // NOTE: do not confuse DOF degree with the polynomial's degree
        if (section->getDegree() >= 2 || (previousSection && previousSection->getDegree() >= 2))
        {
            handle->setDOF(0); // if this or the previous section is quadratic or cubic => no DOF
        }
        else if (previousHandle && (previousHandle->getPosDOF() < 1))
        {
            handle->setDOF(1); // if the previous handle has no DOF, this has one
        }
        else
        {
            handle->setDOF(2);
        }

        // Adjust previous handle //
        //
        if (section->getDegree() >= 2 && previousHandle)
        {
            if (previousHandle->getPosDOF() > 1)
            {
                previousHandle->setDOF(1); // if this handle has no DOF, the previous one should have one (or less)
            }
        }

        // Done //
        //
        previousSection = section;
        previousHandle = handle;
    }

    // Last handle of the road //
    //
    ElevationMoveHandle *handle = new ElevationMoveHandle(elevationEditor_, moveHandles_);
    handle->registerLowSlot(previousSection);
    if (previousHandle->getPosDOF() >= 1)
    {
        handle->setDOF(2);
    }
    else
    {
        handle->setDOF(1);
    }

    // Z depth //
    //
    moveHandles_->setZValue(1.0);
}

void
ElevationRoadPolynomialItem::deleteMoveHandles()
{
    foreach (QGraphicsItem *child, moveHandles_->childItems())
    {
        elevationEditor_->getProfileGraph()->addToGarbage(child);
    }
}

//##################//
// Observer Pattern //
//##################//

void
ElevationRoadPolynomialItem::updateObserver()
{
    // Parent //
    //
    RoadItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Road //
    //
    int changes = getRoad()->getRoadChanges();

    if (changes & RSystemElementRoad::CRD_ElevationSectionChange)
    {
        // A section has been added.
        //
        foreach (ElevationSection *section, getRoad()->getElevationSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                // SectionItem //
                //
                new ElevationSectionPolynomialItem(this, section);
            }
        }

        rebuildMoveHandles();
    }
}
