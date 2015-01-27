/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.04.2010
**
**************************************************************************/

#include "trackrotatehandle.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/commands/trackcommands.hpp"
#include "src/data/visitors/trackmovevalidator.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/editors/trackeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsItem>

// Utils //
//
#include "math.h"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

TrackRotateHandle::TrackRotateHandle(TrackEditor *trackEditor, QGraphicsItem *parent)
    : RotateHandle(parent)
    , lowSlot_(NULL)
    , highSlot_(NULL)
    , rotDOF_(0)
{
    // Editor //
    //
    trackEditor_ = trackEditor;

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Color //
    //
    updateColor();

    // Mouse Move Pos //
    //
    mousePos_ = QPointF(0.0, 0.0);
}

TrackRotateHandle::~TrackRotateHandle()
{
    if (lowSlot_)
    {
        lowSlot_->detachObserver(this);
    }

    if (highSlot_)
    {
        highSlot_->detachObserver(this);
    }

    //	trackEditor_->unregisterTrackRotateHandle(this);
}

//################//
// FUNCTIONS      //
//################//

void
TrackRotateHandle::registerLowSlot(TrackComponent *trackComponent)
{
    lowSlot_ = trackComponent;

    // Observer //
    //
    lowSlot_->attachObserver(this);

    // Transformation //
    //
    if (!highSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
        setPos(lowSlot_->getGlobalPoint(lowSlot_->getSEnd()));
        setRotation(lowSlot_->getGlobalHeading(lowSlot_->getSEnd()));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom //
    //
    updateDOF();
}

void
TrackRotateHandle::registerHighSlot(TrackComponent *trackComponent)
{
    highSlot_ = trackComponent;

    // Observer //
    //
    highSlot_->attachObserver(this);

    // Transformation //
    //
    if (!lowSlot_) // do not set pos twice
    {
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
        setPos(highSlot_->getGlobalPoint(highSlot_->getSStart()));
        setRotation(highSlot_->getGlobalHeading(highSlot_->getSStart()));
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    // Degrees Of Freedom //
    //
    updateDOF();
}

void
TrackRotateHandle::updateDOF()
{
    rotDOF_ = calculateRotDOF();
    updateColor();
}

void
TrackRotateHandle::updateColor()
{
    if (rotDOF_ == 0)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightRed()));
        setPen(QPen(ODD::instance()->colors()->darkRed()));
    }
    else if (rotDOF_ == 1)
    {
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));
    }
}

int
TrackRotateHandle::calculateRotDOF()
{
    // HighSlot //
    //
    if (highSlot_)
    {
        if (highSlot_->getStartRotDOF() == 0)
        {
            return 0; // No Chance.
        }
    }

    // LowSlot //
    //
    if (lowSlot_)
    {
        if (lowSlot_->getEndRotDOF() == 0)
        {
            return 0; // No Chance.
        }
    }

    return 1;
}

//################//
// OBSERVER       //
//################//

/*!
*
* Does not know if highSlot or lowSlot has changed.
*/
void
TrackRotateHandle::updateObserver()
{
    int changes = 0x0;

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
        trackEditor_->getProjectGraph()->addToGarbage(this);
        return;
    }

    if (lowSlot_)
    {
        // TrackComponentChanges //
        //
        changes = lowSlot_->getTrackComponentChanges();

        // Deletion //
        //
        if ((changes & TrackComponent::CTC_TransformChange)
            || (changes & TrackComponent::CTC_ShapeChange)
            || (changes & TrackComponent::CTC_LengthChange))
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(lowSlot_->getGlobalPoint(lowSlot_->getSEnd()));
            setRotation(lowSlot_->getGlobalHeading(lowSlot_->getSEnd()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }

    if (highSlot_)
    {
        // TrackComponentChanges //
        //
        changes = highSlot_->getTrackComponentChanges();

        // Deletion //
        //
        if (changes & TrackComponent::CTC_TransformChange)
        {
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            setPos(highSlot_->getGlobalPoint(highSlot_->getSStart()));
            setRotation(highSlot_->getGlobalHeading(highSlot_->getSStart()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
TrackRotateHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    //	if(change == QGraphicsItem::ItemSelectedHasChanged)
    //	{
    //		if(value.toBool())
    //		{
    //			trackEditor_->registerTrackRotateHandle(this);
    //		}
    //		else
    //		{
    //			trackEditor_->unregisterTrackRotateHandle(this);
    //		}
    //		return value;
    //	}

    /*else */
    if (change == QGraphicsItem::ItemPositionChange)
    {
        if (rotDOF_ == 0)
        {
            return pos(); // no chance
        }

        QPointF direction = mousePos_ - scenePos();
        double heading = (atan2(direction.y(), direction.x()) * 360.0 / (2.0 * M_PI)); // (catches x == 0)
        //double dHeading = (180+atan2(direction.y(), direction.x()) *360.0/(2.0*M_PI)) - heading_; // (catches x == 0)
        //dHeading =  trackEditor_->rotateTrackRotateHandles(dHeading, heading_);

        // Check if rotation is valid //
        //
        TrackMoveValidator *validationVisitor = new TrackMoveValidator();
        validationVisitor->setGlobalHeading(heading);
        if (getHighSlot())
        {
            validationVisitor->setState(TrackMoveValidator::STATE_STARTHEADING);
            getHighSlot()->accept(validationVisitor);
        }
        if (getLowSlot())
        {
            validationVisitor->setState(TrackMoveValidator::STATE_ENDHEADING);
            getLowSlot()->accept(validationVisitor);
        }
        if (validationVisitor->isValid())
        {
            if (lowSlot_ || highSlot_)
            {
                TrackComponentSingleHeadingCommand *command = new TrackComponentSingleHeadingCommand(lowSlot_, highSlot_, heading, NULL);
                if (command->isValid())
                {
                    if (lowSlot_)
                    {
                        lowSlot_->getUndoStack()->push(command);
                    }
                    else /* if(highSlot_) proofed that exists */
                    {
                        highSlot_->getUndoStack()->push(command);
                    }
                }
                else
                {
                    delete command;
                }
            }
        }

        return pos(); // no translation
    }

    return QGraphicsItem::itemChange(change, value);
}

void
TrackRotateHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    mousePos_ = event->scenePos();
    QGraphicsPathItem::mouseMoveEvent(event); // pass to baseclass
}
