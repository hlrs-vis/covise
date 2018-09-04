/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.02.2010
**
**************************************************************************/

#include "lanewidth.hpp"

#include "lanesection.hpp"
#include "lane.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

LaneWidth::LaneWidth(double sOffset, double a, double b, double c, double d)
    : DataElement()
    , Polynomial(a, b, c, d)
    , laneWidthChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
{
}

LaneWidth::~LaneWidth()
{
}

void
LaneWidth::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addLaneWidthChanges(LaneWidth::CLW_ParentLaneChanged);
}

/*! \brief Convenience function. Calls f(sSection) of the Polynomial class.
*/
double
LaneWidth::getWidth(double sSection) const
{
    return f(sSection - sOffset_);
}

double
LaneWidth::getSlope(double sSection) const
{
    return df(sSection - sOffset_);
}

double
LaneWidth::getCurvature(double sSection) const
{
    return ddf(sSection - sOffset_);
}

//####################//
// Width Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneWidth::getSSectionStartAbs() const
{
    return sOffset_ + getParentLane()->getParentLaneSection()->getSStart();
}

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneWidth::getSSectionEnd() const
{
    return parentLane_->getWidthEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneWidth::getLength() const
{
	return parentLane_->getWidthEnd(sOffset_) - getSSectionStartAbs();
}

void
LaneWidth::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addLaneWidthChanges(LaneWidth::CLW_OffsetChanged);
}

double
LaneWidth::getT(double s)
{
	return parentLane_->getParentLaneSection()->getLaneSpanWidth(parentLane_->getId(), 0, s + getParentLane()->getParentLaneSection()->getSStart());
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneWidth::notificationDone()
{
    laneWidthChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneWidth::addLaneWidthChanges(int changes)
{
    if (changes)
    {
        laneWidthChanges_ |= changes;
        notifyObservers();
    }
    if (parentLane_)
        parentLane_->addLaneChanges(Lane::CLN_WidthsChanged);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneWidth *
LaneWidth::getClone()
{
    LaneWidth *clone = new LaneWidth(sOffset_, a_, b_, c_, d_);

    return clone;
}

void
LaneWidth::setParameters(double a, double b, double c, double d)
{
    protectedSetParameters(a, b, c, d);

    addLaneWidthChanges(LaneWidth::CLW_WidthChanged);
}

void
LaneWidth::setOffsetParameters(bool start, const QPointF &dPos)
{

	LaneSection *parentLaneSection = parentLane_->getParentLaneSection();
	RSystemElementRoad *parentRoad = parentLaneSection->getParentRoad();
	QTransform trafo;
	double s = sOffset_ + parentLaneSection->getSStart();
	QPointF pos = parentRoad->getGlobalPoint(s, getWidth(sOffset_));
	double heading = parentRoad->getGlobalHeading(s);
	QVector2D n = parentRoad->getGlobalNormal(s);

	int id = parentLane_->getId();
	double val = QVector2D::dotProduct(QVector2D(dPos), n) / n.length();
	QPointF d = (n.normalized() * val).toPointF(); // Section Border has to be preserved //
	if (id < 0)
	{
		d = -d;
	}
	
	trafo.translate(pos.x(), pos.y());
	trafo.rotate(heading);

	// Change start point //  // !!!!start point is section start point and cannot change s //
	//
	if (start)
	{
		// Local to internal (Parameters are given in internal coordinates) //
		//

		/*double t = getTLength(getSSectionEnd() - sOffset_);
		QPointF t1 = trafo.inverted().map(pos + d);
		QPointF t2 = QPointF(t, f(t));
		QPointF dPosLocal = trafo.inverted().map(pos + d) - QPointF(0.0, f(0.0));
		Polynomial::setParameters(QPointF(t, f(t)) - dPosLocal); */
//		qDebug() << a_ << b_ << c_ << d_;
//		QPointF t1 = trafo.inverted().map(pos + d);
		/*double t = getTLength(getSSectionEnd() - sOffset_);
		QPointF t2 = QPointF(t, f(t));*/
		double sEnd = getSSectionEnd();
		QPointF posEnd = parentRoad->getGlobalPoint(sEnd, getWidth(sEnd - parentLaneSection->getSStart()));
		a_ += val;
		b_ = (posEnd.y() - pos.y()) / (posEnd.x() - pos.x());
//		QPointF t2 = trafo.inverted().map(posEnd);
		//a_ = t1.manhattanLength();
		//a_ += t1.y();
		//b_ = (t2.y() - t1.y()) / (t2.x() - t1.x());
		if (parentLane_->getWidthEntries().size() > 1)
		{
			Polynomial::setParameters(posEnd - pos);
		}
//		qDebug() << t1 << t2;
//		qDebug() << a_ << b_ << c_ << d_;

	//	parentLane_->moveWidthEntry(sOffset_, parentRoad->getSFromGlobalPoint(pos + dPos) - parentLaneSection->getSStart());
	}

	// Change end point //
	//
	else
	{
		qDebug() << "end";
		QPointF end = parentRoad->getGlobalPoint(getSSectionEnd(), getWidth(getSSectionEnd())) + d;
		b_ = (end.y() - pos.y()) / (end.x() - pos.x());
		if ((fabs(c_) > NUMERICAL_ZERO6) || (fabs(d_) > NUMERICAL_ZERO6) || (parentLane_->getWidthEntries().size() > 1))
		{
			Polynomial::setParameters(trafo.inverted().map(end));
		}
	}

	addLaneWidthChanges(LaneWidth::CLW_WidthChanged);

	if (id < 0)
	{
		id--;
		for (id; id >= parentLaneSection->getRightmostLaneId(); id--)
		{
			foreach(LaneWidth *laneWidth, parentLaneSection->getLane(id)->getWidthEntries())
			{
				laneWidth->addLaneWidthChanges(LaneWidth::CLW_WidthChanged);
			}
		}
	}
	else if (id > 0)
	{
		id++;
		for (id; id <= parentLaneSection->getLeftmostLaneId(); id++)
		{
			foreach(LaneWidth *laneWidth, parentLaneSection->getLane(id)->getWidthEntries())
			{
				laneWidth->addLaneWidthChanges(LaneWidth::CLW_WidthChanged);
			}
		}
	} 
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneWidth::accept(Visitor *visitor)
{
    visitor->visit(this);
}
