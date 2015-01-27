/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.06.2010
**
**************************************************************************/

#include "elevationmovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/commands/elevationsectioncommands.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/editors/elevationeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneMouseEvent>

// Utils //
//
#include "math.h"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

ElevationMoveHandle::ElevationMoveHandle(ElevationEditor *elevationEditor, QGraphicsItem *parent)
    : MoveHandle(parent)
    , lowSlot_(NULL)
    , highSlot_(NULL)
    , posDOF_(0)
{
    // Editor //
    //
    elevationEditor_ = elevationEditor;

    // Color //
    //
    updateColor();

    // ContextMenu //
    //
    removeAction_ = getContextMenu()->addAction("Remove Corner");
    smoothAction_ = getContextMenu()->addAction("Smooth");

    connect(removeAction_, SIGNAL(triggered()), this, SLOT(removeCorner()));
    connect(smoothAction_, SIGNAL(triggered()), this, SLOT(smoothCorner()));
}

ElevationMoveHandle::~ElevationMoveHandle()
{
    elevationEditor_->unregisterMoveHandle(this);

    // Observer Pattern //
    //
    if (lowSlot_)
    {
        lowSlot_->detachObserver(this);
    }

    if (highSlot_)
    {
        highSlot_->detachObserver(this);
    }
}

//################//
// FUNCTIONS      //
//################//

void
ElevationMoveHandle::registerLowSlot(ElevationSection *elevationSection)
{
    lowSlot_ = elevationSection;

    // Observer //
    //
    lowSlot_->attachObserver(this);

    // Transformation //
    //
    if (!highSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false); // tmp deactivation
        setPos(elevationSection->getSEnd(), elevationSection->getElevation(elevationSection->getSEnd()));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom Fries //
    //
    //	updateDOF();
}

void
ElevationMoveHandle::registerHighSlot(ElevationSection *elevationSection)
{
    highSlot_ = elevationSection;

    // Observer //
    //
    highSlot_->attachObserver(this);

    // Transformation //
    //
    if (!lowSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
        setPos(elevationSection->getSStart(), elevationSection->getElevation(elevationSection->getSStart()));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom //
    //
    //	updateDOF();
}

void
ElevationMoveHandle::setDOF(int dof)
{
    posDOF_ = dof;
    updateColor();

    // Menu //
    //
    if (posDOF_ == 2)
    {
        //		removeAction_->setEnabled(true);
        smoothAction_->setEnabled(true);
    }
    else if (posDOF_ == 1)
    {
        //		removeAction_->setEnabled(false);
        smoothAction_->setEnabled(true);
    }
    else
    {
        //		removeAction_->setEnabled(false);
        smoothAction_->setEnabled(false);
    }

    removeAction_->setEnabled(true);
}

void
ElevationMoveHandle::updateColor()
{
    if (posDOF_ == 0)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
    }
    else if (posDOF_ == 1)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        setPen(QPen(ODD::instance()->colors()->darkOrange()));
    }
    else if (posDOF_ == 2)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
    }
}

//################//
// OBSERVER       //
//################//

/*!
*
* Does not know if highSlot or lowSlot has changed.
*/
void
ElevationMoveHandle::updateObserver()
{
    int changes = 0x0;

    // Deleted? //
    //
    if (lowSlot_)
    {
        // DataElementChanges //
        //
        changes = lowSlot_->getDataElementChanges();

        // Deletion //
        //
        if ((changes & DataElement::CDE_DataElementDeleted)
            || (changes & DataElement::CDE_DataElementRemoved))
        {
            lowSlot_->detachObserver(this);
            lowSlot_ = NULL;
        }
    }

    if (highSlot_)
    {
        // DataElementChanges //
        //
        changes = highSlot_->getDataElementChanges();

        // Deletion //
        //
        if ((changes & DataElement::CDE_DataElementDeleted)
            || (changes & DataElement::CDE_DataElementRemoved))
        {
            highSlot_->detachObserver(this);
            highSlot_ = NULL;
        }
    }

    if (!lowSlot_ && !highSlot_)
    {
        // No high and no low slot, so delete me please //
        //
        elevationEditor_->getProfileGraph()->addToGarbage(this);
        return;
    }

    // LowSlot //
    //
    if (lowSlot_)
    {
        // ElevationSectionChanges //
        //
        changes = lowSlot_->getElevationSectionChanges();
        int roadSectionChanges = lowSlot_->getRoadSectionChanges();

        // Deletion //
        //
        if ((changes & ElevationSection::CEL_ParameterChange)
            || (roadSectionChanges & RoadSection::CRS_SChange)
            || (roadSectionChanges & RoadSection::CRS_LengthChange))
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(lowSlot_->getSEnd(), lowSlot_->getElevation(lowSlot_->getSEnd()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }

    // HighSlot //
    //
    if (highSlot_)
    {
        // ElevationSectionChanges //
        //
        changes = highSlot_->getElevationSectionChanges();
        int roadSectionChanges = highSlot_->getRoadSectionChanges();

        // Deletion //
        //
        if ((changes & ElevationSection::CEL_ParameterChange)
            || (roadSectionChanges & RoadSection::CRS_SChange)
            || (roadSectionChanges & RoadSection::CRS_LengthChange))
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(highSlot_->getSStart(), highSlot_->getElevation(highSlot_->getSStart()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }
}

//################//
// SLOTS          //
//################//

void
ElevationMoveHandle::removeCorner()
{
    if (!lowSlot_ || !highSlot_)
    {
        return;
    }

    MergeElevationSectionCommand *command = new MergeElevationSectionCommand(lowSlot_, highSlot_, NULL);
    if (command->isValid())
    {
        lowSlot_->getUndoStack()->push(command);
    }
    else
    {
        delete command;
    }
}

void
ElevationMoveHandle::smoothCorner()
{
    if (!lowSlot_ || !highSlot_)
    {
        return;
    }

    SmoothElevationSectionCommand *command = new SmoothElevationSectionCommand(lowSlot_, highSlot_, elevationEditor_->getSmoothRadius(), NULL);
    if (command->isValid())
    {
        lowSlot_->getUndoStack()->push(command);
    }
    else
    {
        elevationEditor_->printStatusBarMsg(command->text(), 4000);
        delete command;
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
ElevationMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            elevationEditor_->registerMoveHandle(this);
        }
        else
        {
            elevationEditor_->unregisterMoveHandle(this);
        }
        return value;
    }

    else if (change == QGraphicsItem::ItemPositionChange)
    {
        return pos(); // no translation
    }

    return MoveHandle::itemChange(change, value);
}

void
ElevationMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let the ElevationEditor handle the movement //
    //
    if (event->modifiers() & Qt::ControlModifier)
    {
        QPointF mousePos = QPointF(floor(event->scenePos().x() + 0.5), floor(event->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
        elevationEditor_->translateMoveHandles(scenePos(), mousePos);
    }
    else
    {
        elevationEditor_->translateMoveHandles(scenePos(), event->scenePos());
    }

    MoveHandle::mouseMoveEvent(event); // pass to baseclass
}

void
ElevationMoveHandle::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    elevationEditor_->getProfileGraph()->postponeGarbageDisposal();
    getContextMenu()->exec(event->screenPos());
    elevationEditor_->getProfileGraph()->finishGarbageDisposal();
}
