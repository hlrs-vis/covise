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

#include "superelevationmovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/commands/superelevationsectioncommands.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/editors/superelevationeditor.hpp"

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

SuperelevationMoveHandle::SuperelevationMoveHandle(SuperelevationEditor *superelevationEditor, QGraphicsItem *parent)
    : MoveHandle(parent)
    , lowSlot_(NULL)
    , highSlot_(NULL)
    , posDOF_(0)
{
    // Editor //
    //
    superelevationEditor_ = superelevationEditor;

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

SuperelevationMoveHandle::~SuperelevationMoveHandle()
{
    superelevationEditor_->unregisterMoveHandle(this);

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
SuperelevationMoveHandle::registerLowSlot(SuperelevationSection *superelevationSection)
{
    lowSlot_ = superelevationSection;

    // Observer //
    //
    lowSlot_->attachObserver(this);

    // Transformation //
    //
    if (!highSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false); // tmp deactivation
        setPos(superelevationSection->getSEnd(), superelevationSection->getSuperelevationDegrees(superelevationSection->getSEnd()));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom Fries //
    //
    //	updateDOF();
}

void
SuperelevationMoveHandle::registerHighSlot(SuperelevationSection *superelevationSection)
{
    highSlot_ = superelevationSection;

    // Observer //
    //
    highSlot_->attachObserver(this);

    // Transformation //
    //
    if (!lowSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
        setPos(superelevationSection->getSStart(), superelevationSection->getSuperelevationDegrees(superelevationSection->getSStart()));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom //
    //
    //	updateDOF();
}

void
SuperelevationMoveHandle::setDOF(int dof)
{
    posDOF_ = dof;
    updateColor();

    // Menu //
    //
    if (posDOF_ == 2)
    {
        removeAction_->setEnabled(true);
        smoothAction_->setEnabled(true);
    }
    else if (posDOF_ == 1)
    {
        removeAction_->setEnabled(false);
        smoothAction_->setEnabled(true);
    }
    else
    {
        removeAction_->setEnabled(false);
        smoothAction_->setEnabled(false);
    }
}

void
SuperelevationMoveHandle::updateColor()
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
SuperelevationMoveHandle::updateObserver()
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
        superelevationEditor_->getProfileGraph()->addToGarbage(this);
        return;
    }

    // LowSlot //
    //
    if (lowSlot_)
    {
        // SuperelevationSectionChanges //
        //
        changes = lowSlot_->getSuperelevationSectionChanges();
        int roadSectionChanges = lowSlot_->getRoadSectionChanges();

        // Deletion //
        //
        if ((changes & SuperelevationSection::CSE_ParameterChange)
            || (roadSectionChanges & RoadSection::CRS_SChange)
            || (roadSectionChanges & RoadSection::CRS_LengthChange))
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(lowSlot_->getSEnd(), lowSlot_->getSuperelevationDegrees(lowSlot_->getSEnd()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }

    // HighSlot //
    //
    if (highSlot_)
    {
        // SuperelevationSectionChanges //
        //
        changes = highSlot_->getSuperelevationSectionChanges();
        int roadSectionChanges = highSlot_->getRoadSectionChanges();

        // Deletion //
        //
        if ((changes & SuperelevationSection::CSE_ParameterChange)
            || (roadSectionChanges & RoadSection::CRS_SChange)
            || (roadSectionChanges & RoadSection::CRS_LengthChange))
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(highSlot_->getSStart(), highSlot_->getSuperelevationDegrees(highSlot_->getSStart()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }
}

//################//
// SLOTS          //
//################//

void
SuperelevationMoveHandle::removeCorner()
{
    if (!lowSlot_ || !highSlot_)
    {
        return;
    }

    MergeSuperelevationSectionCommand *command = new MergeSuperelevationSectionCommand(lowSlot_, highSlot_, NULL);
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
SuperelevationMoveHandle::smoothCorner()
{
    if (!lowSlot_ || !highSlot_)
    {
        return;
    }

    SmoothSuperelevationSectionCommand *command = new SmoothSuperelevationSectionCommand(lowSlot_, highSlot_, superelevationEditor_->getSmoothRadius(), NULL);
    if (command->isValid())
    {
        lowSlot_->getUndoStack()->push(command);
    }
    else
    {
        superelevationEditor_->printStatusBarMsg(command->text(), 4000);
        delete command;
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
SuperelevationMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            superelevationEditor_->registerMoveHandle(this);
        }
        else
        {
            superelevationEditor_->unregisterMoveHandle(this);
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
SuperelevationMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let the SuperelevationEditor handle the movement //
    //
    if (event->modifiers() & Qt::ControlModifier)
    {
        QPointF mousePos = QPointF(floor(event->scenePos().x() + 0.5), floor(event->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
        superelevationEditor_->translateMoveHandles(scenePos(), mousePos);
    }
    else
    {
        superelevationEditor_->translateMoveHandles(scenePos(), event->scenePos());
    }

    MoveHandle::mouseMoveEvent(event); // pass to baseclass
}
