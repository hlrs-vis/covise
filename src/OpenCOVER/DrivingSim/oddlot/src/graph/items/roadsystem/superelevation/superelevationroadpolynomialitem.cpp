/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#include "superelevationroadpolynomialitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/profilegraph.hpp"

#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "superelevationsectionpolynomialitem.hpp"
#include "superelevationmovehandle.hpp"

// Editor //
//
#include "src/graph/editors/superelevationeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsScene>

SuperelevationRoadPolynomialItem::SuperelevationRoadPolynomialItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , superelevationEditor_(NULL)
    , moveHandles_(NULL)
{
    init();
}

SuperelevationRoadPolynomialItem::~SuperelevationRoadPolynomialItem()
{
    // Do not use the garbage in destructors (disposal is already running)
}

void
SuperelevationRoadPolynomialItem::init()
{
    // SuperelevationEditor //
    //
    superelevationEditor_ = dynamic_cast<SuperelevationEditor *>(getProfileGraph()->getProjectWidget()->getProjectEditor());
    if (!superelevationEditor_)
    {
        qDebug("Warning 1006241105! SuperelevationRoadPolynomialItem not created by an SuperelevationEditor");
    }

    foreach (SuperelevationSection *section, getRoad()->getSuperelevationSections())
    {
        // SectionItem //
        //
        new SuperelevationSectionPolynomialItem(this, section);
    }

    // MoveHandles //
    //
    moveHandles_ = new QGraphicsPathItem(this);
    rebuildMoveHandles();
}

QRectF
SuperelevationRoadPolynomialItem::boundingRect() const
{
    QRectF boundingBox;
    foreach (QGraphicsItem *childItem, childItems())
    {
        boundingBox = boundingBox.united(childItem->boundingRect());
    }
    return boundingBox;
}
void
SuperelevationRoadPolynomialItem::rebuildMoveHandles()
{
    deleteMoveHandles();

    // Have fun understanding this //
    //
    SuperelevationSection *previousSection = NULL;
    SuperelevationMoveHandle *previousHandle = NULL;
    foreach (SuperelevationSection *section, getRoad()->getSuperelevationSections())
    {
        SuperelevationMoveHandle *handle = new SuperelevationMoveHandle(superelevationEditor_, moveHandles_);
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
    SuperelevationMoveHandle *handle = new SuperelevationMoveHandle(superelevationEditor_, moveHandles_);
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
SuperelevationRoadPolynomialItem::deleteMoveHandles()
{
    foreach (QGraphicsItem *child, moveHandles_->childItems())
    {
        superelevationEditor_->getProfileGraph()->addToGarbage(child);
    }
}

//##################//
// Observer Pattern //
//##################//

void
SuperelevationRoadPolynomialItem::updateObserver()
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

    if (changes & RSystemElementRoad::CRD_SuperelevationSectionChange)
    {
        // A section has been added.
        //
        foreach (SuperelevationSection *section, getRoad()->getSuperelevationSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                // SectionItem //
                //
                new SuperelevationSectionPolynomialItem(this, section);
            }
        }

        rebuildMoveHandles();
    }
}
