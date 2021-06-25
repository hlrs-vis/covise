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

#include "laneborderhandle.hpp"

// Data //
//
#include "src/data/roadsystem/sections/laneborder.hpp"

// Graph //
//
#include "src/graph/editors/laneeditor.hpp"


//################//
// CONSTRUCTOR    //
//################//

LaneBorderHandle::LaneBorderHandle(LaneEditor *laneEditor, QGraphicsItem *parent)
    : LaneMoveHandle<LaneBorder>(laneEditor, parent)
{

    // ContextMenu //
    //
    removeAction_ = getContextMenu()->addAction("Remove Corner");
    smoothAction_ = getContextMenu()->addAction("Smooth");

    connect(removeAction_, SIGNAL(triggered()), this, SLOT(removeCorner()));
    connect(smoothAction_, SIGNAL(triggered()), this, SLOT(smoothCorner()));
}

LaneBorderHandle::~LaneBorderHandle()
{
}

//################//
// SLOTS          //
//################//

void
LaneBorderHandle::removeCorner()
{
    if (!getLowSlot() || !getHighSlot())
    {
        return;
    }

    /*	MergeLaneBorderSectionCommand * command = new MergeLaneBorderSectionCommand(lowSlot_, highSlot_, NULL);
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
LaneBorderHandle::smoothCorner()
{
    if (!getLowSlot() || !getHighSlot())
    {
        return;
    }
    //TODO
    /*	SmoothLaneBorderSectionCommand * command = new SmoothLaneBorderSectionCommand(lowSlot_, highSlot_, laneWidthEditor_->getSmoothRadius(), NULL);
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

// ################//
// EVENTS         //
//################//

QVariant
LaneBorderHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
	// NOTE: position is relative to parent!!! //
	//
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		if (value.toBool())
		{
			laneEditor_->registerMoveHandle(this);
		}
		else
		{
			laneEditor_->unregisterMoveHandle(this);
		}
		return value;
	}

	else if (change == QGraphicsItem::ItemPositionChange)
	{
		return pos(); // no translation
	}

	return MoveHandle::itemChange(change, value);
}





