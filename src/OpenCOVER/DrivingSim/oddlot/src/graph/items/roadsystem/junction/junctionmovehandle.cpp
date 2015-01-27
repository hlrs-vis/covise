/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   20.04.2010
**
**************************************************************************/

#include "junctionmovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/commands/trackcommands.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
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

JunctionMoveHandle::JunctionMoveHandle(JunctionEditor *junctionEditor, QGraphicsItem *parent)
    : MoveHandle(parent)
    , lowSlot_(NULL)
    , highSlot_(NULL)
    , posDOF_(0)
    , posDOFHeading_(0.0)
{
    // Editor //
    //
    junctionEditor_ = junctionEditor;

    // Color //
    //
    updateColor();
}

JunctionMoveHandle::~JunctionMoveHandle()
{
    if (lowSlot_)
    {
        lowSlot_->detachObserver(this);
    }

    if (highSlot_)
    {
        highSlot_->detachObserver(this);
    }

    junctionEditor_->unregisterJunctionMoveHandle(this);
}

//################//
// FUNCTIONS      //
//################//

void
JunctionMoveHandle::registerLowSlot(TrackComponent *trackComponent)
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

    // Degrees Of Freedom Fries //
    //
    updateDOF();
}

void
JunctionMoveHandle::registerHighSlot(TrackComponent *trackComponent)
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
JunctionMoveHandle::updateDOF()
{
    posDOF_ = calculatePosDOF();
    updateColor();
}

void
JunctionMoveHandle::updateColor()
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

int
JunctionMoveHandle::calculatePosDOF()
{
    // Init //
    //
    int dof = 2;

    // HighSlot //
    //
    if (highSlot_)
    {
        dof = highSlot_->getStartPosDOF();
        if (dof == 0)
        {
            return 0; // No Chance.
        }
        else if (dof == 1)
        {
            posDOFHeading_ = highSlot_->getGlobalHeading(highSlot_->getSStart());
        }
    }

    // LowSlot //
    //
    if (lowSlot_)
    {
        int lowDof = lowSlot_->getEndPosDOF();
        if (lowDof == 2)
        {
            return dof; // highSlot: dof, lowSlot: 2
        }
        else if (lowDof == 1)
        {
            if (dof == 2)
            {
                posDOFHeading_ = lowSlot_->getGlobalHeading(lowSlot_->getSEnd());
                return 1; // highSlot: 2, lowSlot: 1
            }
            else if (dof == 1)
            {
                // Check //
                //
                // NOTE: it would be better to let the TrackComponents decide in which direction they can be modified!
                // if dof == 1: there is a highSlot! so no more checking is needed!
                if (fabs(highSlot_->getGlobalHeading(highSlot_->getSStart()) - lowSlot_->getGlobalHeading(lowSlot_->getSEnd())) < NUMERICAL_ZERO4)
                {
                    // Heading is equal //
                    //
                    posDOFHeading_ = highSlot_->getGlobalHeading(highSlot_->getSStart());
                    return 1; // highSlot: 1, lowSlot: 1 (equal)
                }
                else
                {
                    return 0; // highSlot: 1, lowSlot: 1 (not equal)
                }
            }
        }
        else /* if(lowDof == 0)*/
        {
            return 0; // highSlot: dof, lowSlot: 0
        }
    }
    return dof;
}

//################//
// OBSERVER       //
//################//

/*!
*
* Does not know if highSlot or lowSlot has changed.
*/
void
JunctionMoveHandle::updateObserver()
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
        junctionEditor_->getProjectGraph()->addToGarbage(this);
        return;
    }

    // LowSlot //
    //
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
            posDOFHeading_ = lowSlot_->getGlobalHeading(lowSlot_->getSEnd());
            setPos(lowSlot_->getGlobalPoint(lowSlot_->getSEnd()));
            setRotation(posDOFHeading_);
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }
    }

    // HighSlot //
    //
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
            posDOFHeading_ = highSlot_->getGlobalHeading(highSlot_->getSStart());
            setPos(highSlot_->getGlobalPoint(highSlot_->getSStart()));
            setRotation(posDOFHeading_);
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
JunctionMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            junctionEditor_->registerJunctionMoveHandle(this);
        }
        else
        {
            junctionEditor_->unregisterJunctionMoveHandle(this);
        }
        return value;
    }

    return MoveHandle::itemChange(change, value);
}

void
JunctionMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let the junctionEditor handle the movement //
    //
    if (isSelected()) // only if this handle is actually selected
    {
        junctionEditor_->translateJunctionMoveHandles(scenePos(), event->scenePos());
    }

    //	MoveHandle::mouseMoveEvent(event); // pass to baseclass
}
