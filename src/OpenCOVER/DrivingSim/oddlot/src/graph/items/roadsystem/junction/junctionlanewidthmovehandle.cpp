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

#include "junctionlanewidthmovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
//#include "src/data/commands/lanewidthcommands.hpp"

// Graph //
//
#include "src/graph/profilegraph.hpp"
#include "src/graph/editors/junctioneditor.hpp"

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

JunctionLaneWidthMoveHandle::JunctionLaneWidthMoveHandle(JunctionEditor *junctionEditor, QGraphicsItem *parent)
    : MoveHandle(parent)
    , lowSlot_(NULL)
    , highSlot_(NULL)
    , posDOF_(0)
{
    // Editor //
    //
    junctionEditor_ = junctionEditor;

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

JunctionLaneWidthMoveHandle::~JunctionLaneWidthMoveHandle()
{
    if (junctionEditor_)
        junctionEditor_->unregisterLaneMoveHandle(this);

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
JunctionLaneWidthMoveHandle::registerLowSlot(LaneWidth *laneWidthSection)
{
    lowSlot_ = laneWidthSection;

    // Observer //
    //
    lowSlot_->attachObserver(this);

    // Transformation //
    //
    if (!highSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false); // tmp deactivation
        double end = laneWidthSection->getSSectionEnd(); // end is in global coordinates, not local ones ...
        setPos(end, laneWidthSection->getWidth(end));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom Fries //
    //
    //	updateDOF();
}

void
JunctionLaneWidthMoveHandle::registerHighSlot(LaneWidth *laneWidthSection)
{
    highSlot_ = laneWidthSection;

    // Observer //
    //
    highSlot_->attachObserver(this);

    // Transformation //
    //
    if (!lowSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
        double start = laneWidthSection->getSSectionStartAbs();
        setPos(start, laneWidthSection->getWidth(start));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom //
    //
    //	updateDOF();
}

void
JunctionLaneWidthMoveHandle::setDOF(int dof)
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
JunctionLaneWidthMoveHandle::updateColor()
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
JunctionLaneWidthMoveHandle::updateObserver()
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
        junctionEditor_->getProfileGraph()->addToGarbage(this);
        return;
    }

    // LowSlot //
    //
    if (lowSlot_)
    {
        // LaneWidthSectionChanges //

        changes = lowSlot_->getLaneWidthChanges();

        //
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(lowSlot_->getSSectionEnd(), lowSlot_->getWidth(lowSlot_->getSSectionEnd()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }

    // HighSlot //
    //
    if (highSlot_)
    {
        // LaneWidthSectionChanges //

        changes = highSlot_->getLaneWidthChanges();

        //
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(highSlot_->getSSectionStartAbs(), highSlot_->getWidth(highSlot_->getSSectionStartAbs()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }
    /*	if(highSlot_)
	{
		highSlot_->getParentLane()->
	}
	else if(lowSlot_)
	{
	}*/
}

//################//
// SLOTS          //
//################//

void
JunctionLaneWidthMoveHandle::removeCorner()
{
    if (!lowSlot_ || !highSlot_)
    {
        return;
    }

    /*	MergeLaneWidthSectionCommand * command = new MergeLaneWidthSectionCommand(lowSlot_, highSlot_, NULL);
	if(command->isValid())
	{
		lowSlot_->getUndoStack()->push(command);
	}
	else
	{
		delete command;
	}
	*/
}

void
JunctionLaneWidthMoveHandle::smoothCorner()
{
    if (!lowSlot_ || !highSlot_)
    {
        return;
    }
    //TODO
    /*	SmoothLaneWidthSectionCommand * command = new SmoothLaneWidthSectionCommand(lowSlot_, highSlot_, laneWidthEditor_->getSmoothRadius(), NULL);
	if(command->isValid())
	{
		lowSlot_->getUndoStack()->push(command);
	}
	else
	{
		laneWidthEditor_->printStatusBarMsg(command->text(), 4000);
		delete command;
	}*/
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
JunctionLaneWidthMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            junctionEditor_->registerLaneMoveHandle(this);
        }
        else
        {
            junctionEditor_->unregisterLaneMoveHandle(this);
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
JunctionLaneWidthMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let the JunctionLaneWidthEditor handle the movement //
    //
    if (event->modifiers() & Qt::ControlModifier)
    {
        QPointF mousePos = QPointF(floor(event->scenePos().x() + 0.5), floor(event->scenePos().y() + 0.5)); // rounded = floor(x+0.5)
        junctionEditor_->translateLaneMoveHandles(scenePos(), mousePos);
    }
    else
    {
        junctionEditor_->translateLaneMoveHandles(scenePos(), event->scenePos());
    }

    MoveHandle::mouseMoveEvent(event); // pass to baseclass
}
