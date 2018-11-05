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

#ifndef LANEMOVEHANDLE_HPP
#define LANEMOVEHANDLE_HPP

#include "src/graph/items/handles/baselanemovehandle.hpp"

// Data  //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/laneborder.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/commands/lanesectioncommands.hpp"



// Graph //
//
#include "src/graph/editors/laneeditor.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

// Qt //
//
#include <QPointF>
#include <QGraphicsItem>

template<typename T, typename U>
class LaneMoveHandle : public BaseLaneMoveHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneMoveHandle(LaneEditor *laneEditor, QGraphicsItem *parent)
        : BaseLaneMoveHandle(laneEditor, parent)
        , lowSlot_(NULL)
        , highSlot_(NULL)

    {
        // Editor //
        //
        laneEditor_ = laneEditor;

        // Color //
        //
        updateColor();

    }

    virtual ~LaneMoveHandle()
    {
        if (laneEditor_)
        {
            laneEditor_->unregisterMoveHandle(this);
        }

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

    virtual T *getLowSlot()
    {
        return lowSlot_;
    }

    virtual U *getHighSlot()
    {
        return highSlot_;
    }

    void registerLowSlot(T *laneBorderSection)
    {
        lowSlot_ = laneBorderSection;

        // Observer //
        //
        lowSlot_->attachObserver(this);

        if (fabs(lowSlot_->getLength() - lowSlot_->getParentLane()->getParentLaneSection()->getSEnd()) < NUMERICAL_ZERO6)
        {
			if (getContextMenu()->actions().size() > 2)
			{
				QAction *removeAction = getContextMenu()->actions().at(0);
				getContextMenu()->removeAction(removeAction);
			}
        }

        // Transformation //
        //
        if (!highSlot_) // do not set pos twice
        {

            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false); // tmp deactivation
            RSystemElementRoad *parentRoad = lowSlot_->getParentLane()->getParentLaneSection()->getParentRoad();
            if (lowSlot_->getParentLane()->getId() > 0)
            {
                setPos(parentRoad->getGlobalPoint(lowSlot_->getSSectionEnd(), lowSlot_->getT(lowSlot_->getSSectionStart() + lowSlot_->getLength())));
            }
            else
            {
                setPos(parentRoad->getGlobalPoint(lowSlot_->getSSectionEnd(), -lowSlot_->getT(lowSlot_->getSSectionStart() + lowSlot_->getLength())));
            }
            setRotation(parentRoad->getGlobalHeading(lowSlot_->getSSectionEnd()));
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
        }


        updateColor();
    }


    void registerHighSlot(U *laneBorderSection)
    {
        highSlot_ = laneBorderSection;

        // Observer //
        //
        highSlot_->attachObserver(this);

        if (highSlot_->getParentLane()->getWidthEntry(0.0) == highSlot_)
        {
			if (getContextMenu()->actions().size() > 2)
			{
				QAction *removeAction = getContextMenu()->actions().at(0);
				getContextMenu()->removeAction(removeAction);
			}
        }

        // Transformation //
        //
        if (!lowSlot_) // do not set pos twice
        {

            setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
            RSystemElementRoad *parentRoad = highSlot_->getParentLane()->getParentLaneSection()->getParentRoad();
            if (highSlot_->getParentLane()->getId() > 0)
            {
                setPos(parentRoad->getGlobalPoint(highSlot_->getSSectionStartAbs(), highSlot_->getT(highSlot_->getSSectionStart())));
            }
            else
            {
                setPos(parentRoad->getGlobalPoint(highSlot_->getSSectionStartAbs(), -highSlot_->getT(highSlot_->getSSectionStart())));
            }
            setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

        }

        updateColor();

    }

    T *getLowSlot() const
    {
        return lowSlot_;
    }
    U *getHighSlot() const
    {
        return highSlot_;
    }

    virtual void removeCorner()
    {

        if (!lowSlot_ || !highSlot_)
        {
            return;
        }


        MergeLaneWidthSectionCommand<T, U> *command = new MergeLaneWidthSectionCommand<T, U>(lowSlot_, highSlot_, NULL);
        if (command->isValid())
        {
            lowSlot_->getUndoStack()->push(command);
        } 
        else
        {
            delete command;
        } 

    }

    virtual void smoothCorner()
    {
        if (!isCorner_)
        {
            return;
        }

        if (!lowSlot_ || !highSlot_)
        {
            return;
        }

        LaneBorderCornerCommand *command = new LaneBorderCornerCommand(lowSlot_, highSlot_, true, NULL);
        laneEditor_->getTopviewGraph()->executeCommand(command);
    }

    virtual void corner()
    {
        if (isCorner_ )
        {
            return;
        }

        if (!lowSlot_ || !highSlot_)
        {
            isCorner_ = true;
            return;
        }

        LaneBorderCornerCommand *command = new LaneBorderCornerCommand(lowSlot_, highSlot_, true,  NULL);
        laneEditor_->getTopviewGraph()->executeCommand(command);

    }


    // Observer Pattern //
    //
    virtual void updateObserver()
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
            // No high and no low slot, so will be deleted //
            //

            return;
        }

        // LowSlot //
        //
        if (lowSlot_)
        {
            // LaneBorderSectionChanges //

            changes = lowSlot_->getLaneWidthChanges();

            if (changes & LaneWidth::CLW_WidthChanged)
            {
                RSystemElementRoad *parentRoad = lowSlot_->getParentLane()->getParentLaneSection()->getParentRoad();
                setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
                if (lowSlot_->getParentLane()->getId() > 0)
                {
                    setPos(parentRoad->getGlobalPoint(lowSlot_->getSSectionEnd(), lowSlot_->getT(lowSlot_->getSSectionStart() + lowSlot_->getLength())));
                }
                else
                {
                    setPos(parentRoad->getGlobalPoint(lowSlot_->getSSectionEnd(), -lowSlot_->getT(lowSlot_->getSSectionStart() + lowSlot_->getLength())));
                }
                setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

            }

            if (changes & LaneWidth::CLW_GradientChanged)
            {
                updateColor();
            }

        }

        // HighSlot //
        //
        if (highSlot_)
        {
            // LaneBorderSectionChanges //

            changes = highSlot_->getLaneWidthChanges();

            if (changes & LaneWidth::CLW_WidthChanged)
            {
                setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
                RSystemElementRoad *parentRoad = highSlot_->getParentLane()->getParentLaneSection()->getParentRoad();
                if (highSlot_->getParentLane()->getId() > 0)
                {
                    setPos(parentRoad->getGlobalPoint(highSlot_->getSSectionStartAbs(), highSlot_->getT(highSlot_->getSSectionStart())));
                }
                else
                {
                    setPos(parentRoad->getGlobalPoint(highSlot_->getSSectionStartAbs(), -highSlot_->getT(highSlot_->getSSectionStart())));
                }
                setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

            }
        }

    }

protected:
    virtual QVariant itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
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

	virtual const QString getText()
	{
		// Text //
		//
		QString text;

		if (highSlot_)
		{
			text = QString("%1").arg(highSlot_->f(0.0), 0, 'f', 2);
		}
		else if (lowSlot_)
		{
			text = QString("%1").arg(lowSlot_->f(lowSlot_->getSSectionEnd() - lowSlot_->getSSectionStartAbs()), 0, 'f', 2);
		}

		return text;
	}

private:
    LaneMoveHandle(); /* not allowed */
    LaneMoveHandle(const LaneMoveHandle &); /* not allowed */
    LaneMoveHandle &operator=(const LaneMoveHandle &); /* not allowed */

    void updateColor()
    {
        if (lowSlot_ && highSlot_)
        {
            if (fabs(lowSlot_->df(highSlot_->getSSectionStartAbs() - lowSlot_->getSSectionStartAbs()) - highSlot_->df(0.0)) < NUMERICAL_ZERO3)
            {
                    isCorner_ = false;
                    setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
                    setPen(QPen(ODD::instance()->colors()->darkOrange()));

                    return;
            }
        }


        isCorner_ = true;
        setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
        setPen(QPen(ODD::instance()->colors()->darkGreen()));

    }



    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    LaneEditor *laneEditor_;

    T *lowSlot_;
    U *highSlot_;

    bool isCorner_;


};


#endif // LANEMOVEHANDLE_HPP
