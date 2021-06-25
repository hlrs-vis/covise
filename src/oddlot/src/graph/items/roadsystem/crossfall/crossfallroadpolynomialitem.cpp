/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.07.2010
**
**************************************************************************/

#include "crossfallroadpolynomialitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/profilegraph.hpp"

#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "crossfallsectionpolynomialitem.hpp"
#include "crossfallmovehandle.hpp"

// Editor //
//
#include "src/graph/editors/crossfalleditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsScene>

CrossfallRoadPolynomialItem::CrossfallRoadPolynomialItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , crossfallEditor_(NULL)
    , moveHandles_(NULL)
{
    init();
}

CrossfallRoadPolynomialItem::~CrossfallRoadPolynomialItem()
{
    // Do not use the garbage in destructors (disposal is already running)
}

void
CrossfallRoadPolynomialItem::init()
{
    // CrossfallEditor //
    //
    crossfallEditor_ = dynamic_cast<CrossfallEditor *>(getProfileGraph()->getProjectWidget()->getProjectEditor());
    if (!crossfallEditor_)
    {
        qDebug("Warning 1006241105! CrossfallRoadPolynomialItem not created by an CrossfallEditor");
    }

    foreach (CrossfallSection *section, getRoad()->getCrossfallSections())
    {
        // SectionItem //
        //
        new CrossfallSectionPolynomialItem(this, section);
    }

    // MoveHandles //
    //
    moveHandles_ = new QGraphicsPathItem(this);
    rebuildMoveHandles();
}

QRectF
CrossfallRoadPolynomialItem::boundingRect() const
{
    QRectF boundingBox;
    foreach (QGraphicsItem *childItem, childItems())
    {
        boundingBox = boundingBox.united(childItem->boundingRect());
    }
    return boundingBox;
}
void
CrossfallRoadPolynomialItem::rebuildMoveHandles()
{
    deleteMoveHandles();

    // Have fun understanding this //
    //
    CrossfallSection *previousSection = NULL;
    CrossfallMoveHandle *previousHandle = NULL;
    foreach (CrossfallSection *section, getRoad()->getCrossfallSections())
    {
        CrossfallMoveHandle *handle = new CrossfallMoveHandle(crossfallEditor_, moveHandles_);
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
    CrossfallMoveHandle *handle = new CrossfallMoveHandle(crossfallEditor_, moveHandles_);
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
CrossfallRoadPolynomialItem::deleteMoveHandles()
{
    foreach (QGraphicsItem *child, moveHandles_->childItems())
    {
        crossfallEditor_->getProfileGraph()->addToGarbage(child);
    }
}

//##################//
// Observer Pattern //
//##################//

void
CrossfallRoadPolynomialItem::updateObserver()
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

    if (changes & RSystemElementRoad::CRD_CrossfallSectionChange)
    {
        // A section has been added.
        //
        foreach (CrossfallSection *section, getRoad()->getCrossfallSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                // SectionItem //
                //
                new CrossfallSectionPolynomialItem(this, section);
            }
        }

        rebuildMoveHandles();
    }
}
