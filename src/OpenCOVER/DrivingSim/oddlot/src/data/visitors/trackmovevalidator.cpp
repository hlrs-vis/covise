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

#include "trackmovevalidator.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Qt //
//
#include <QVector2D>

// Utils //
//
#include "math.h"

/*!
*
*/
TrackMoveValidator::TrackMoveValidator()
    : isValid_(true)
    , state_(TrackMoveValidator::STATE_STARTPOINT)
    , dPos_(0.0, 0.0)
    , heading_(0.0)

{
}

void
TrackMoveValidator::setState(TrackMoveValidator::State state)
{
    state_ = state;
}

void
TrackMoveValidator::setGlobalDeltaPos(const QPointF &dPos)
{
    dPos_ = dPos;
}

void
TrackMoveValidator::setGlobalHeading(const double heading)
{
    heading_ = heading;

    while (heading_ < 0.0)
    {
        heading_ += 360.0;
    }
    while (heading_ >= 360.0)
    {
        heading_ -= 360.0;
    }
}

/*!
*
*/
void
TrackMoveValidator::visit(TrackSpiralArcSpiral *trackComposite)
{
    SpArcSParameters *parameters = trackComposite->getClonedParameters();

    // Change start point //
    //
    if (state_ == TrackMoveValidator::STATE_STARTPOINT)
    {
        // Global to internal (Parameters are given in internal coordinates) //
        //
        // 1.) New global start point: trackComposite->getGlobalPoint(trackComposite->getSStart()) + dPos_
        // 2.) Convert to internal coordinates
        // 3.) Calculate deltaPos (in internal coordinates)
        // Alternative: calculate a transform matrix WITHOUT translation.
        QPointF deltaPos(trackComposite->getGlobalTransform().inverted().map(trackComposite->getGlobalPoint(trackComposite->getSStart()) + dPos_) - trackComposite->getPoint(trackComposite->getSStart()));

        // Set parameter //
        //
        parameters->setEndPoint(parameters->getEndPoint() - deltaPos);
    }

    // Change end point //
    //
    else if (state_ == TrackMoveValidator::STATE_ENDPOINT)
    {
        // Global to internal (Parameters are given in internal coordinates) //
        //
        QPointF deltaPos(trackComposite->getGlobalTransform().inverted().map(trackComposite->getGlobalPoint(trackComposite->getSEnd()) + dPos_) - trackComposite->getPoint(trackComposite->getSEnd()));

        // Set parameter //
        //
        parameters->setEndPoint(parameters->getEndPoint() + deltaPos);
    }

    // Change start heading //
    //
    else if (state_ == TrackMoveValidator::STATE_STARTHEADING)
    {
        // Global to internal (Parameters are given in internal coordinates) //
        //
        double heading = heading_;
        heading = heading - trackComposite->getGlobalHeading(trackComposite->getSStart());
        double deltaHeading = heading /* - 0.0*/;

        QTransform trafo;
        trafo.rotate(deltaHeading);

        // Set parameter //
        //
        parameters->setEndHeadingDeg(parameters->getEndHeadingRad() * 360.0 / (2.0 * M_PI) - deltaHeading);
        parameters->setEndPoint(trafo.inverted().map(parameters->getEndPoint()));
    }

    // Change end heading //
    //
    else if (state_ == TrackMoveValidator::STATE_ENDHEADING)
    {
        // Global to internal (Parameters are given in internal coordinates) //
        //
        double heading = heading_;
        heading = heading - trackComposite->getGlobalHeading(trackComposite->getSStart());

        // Set parameter //
        //
        parameters->setEndHeadingDeg(heading);
    }

    // Check //
    //
    if (!parameters->isValid())
    {
        isValid_ = false;
    }

    // Clean up //
    //
    delete parameters;

    return;
}

/*! \brief Checks if the line will still be longer than one [mm] (and thus also not negative).
*
*/
void
TrackMoveValidator::visit(TrackElementLine *trackElement)
{
    // Check heading to keep the new point on the track //
    //
    /*	RSystemElementRoad * parentRoad = trackElement->getParentRoad();
	TrackComponent * track = NULL;
	if ((state_ == TrackMoveValidator::STATE_STARTPOINT) || (state_ == STATE_STARTHEADING))
	{
		track = parentRoad->getTrackComponentBefore(trackElement->getSStart());
	}
	else
	{
		track = parentRoad->getTrackComponent(trackElement->getSEnd());
	}
	TrackElementLine * trackLine = dynamic_cast<TrackElementLine *>(track); 

	if (trackLine || ((((state_ == TrackMoveValidator::STATE_STARTPOINT) || (state_ == STATE_STARTHEADING)) && (abs(trackElement->getSStart()) < NUMERICAL_ZERO3)) || 
		(((state_ == TrackMoveValidator::STATE_ENDPOINT) || (state_ == STATE_ENDHEADING))&& (abs(trackElement->getSEnd() - parentRoad->getLength()) < NUMERICAL_ZERO3))))
	{
		double angle = atan2(dPos_.y(), dPos_.x());
		if(angle < 0.0)
		{
			angle += 2.0*M_PI;
		}
		double angle2 = atan2(-dPos_.y(), -dPos_.x());
		if(angle2 < 0.0)
		{
			angle2 += 2.0*M_PI;
		}
		if((fabs(trackElement->getGlobalHeadingRad(trackElement->getSStart()) - angle) > NUMERICAL_ZERO4)
			&&	(fabs(trackElement->getGlobalHeadingRad(trackElement->getSStart()) - angle2) > NUMERICAL_ZERO4)
			)
		{
			isValid_ = false;
			return;
		} 
	}*/

    // Change start point //
    //
    if (state_ == TrackMoveValidator::STATE_STARTPOINT)
    {
        QVector2D t = trackElement->getGlobalTangent(trackElement->getSStart()); // unit vector
        QVector2D l = QVector2D(trackElement->getGlobalPoint(trackElement->getSEnd()) - (dPos_ + trackElement->getGlobalPoint(trackElement->getSStart())));
        if (QVector2D::dotProduct(t, l) <= NUMERICAL_ZERO3) // less than one millimeter or negative
        {
            isValid_ = false;
        }
    }

    // Change end point //
    //
    else if (state_ == TrackMoveValidator::STATE_ENDPOINT)
    {
        QVector2D t = trackElement->getGlobalTangent(trackElement->getSStart()); // unit vector
        QVector2D l = QVector2D((dPos_ + trackElement->getGlobalPoint(trackElement->getSEnd())) - trackElement->getGlobalPoint(trackElement->getSStart()));
        if (QVector2D::dotProduct(t, l) <= NUMERICAL_ZERO3) // less than one millimeter or negative
        {
            isValid_ = false; // new position is not on the track line
        }
    }

    return;
}

/*! \brief There are currently no arc modifications supported.
*
*/
void
TrackMoveValidator::visit(TrackElementArc * /*trackElement*/)
{
    isValid_ = false;
    return;
}

/*! \brief There are currently no spiral modifications supported.
*
*/
void
TrackMoveValidator::visit(TrackElementSpiral * /*trackElement*/)
{
    isValid_ = false;
    return;
}

void
TrackMoveValidator::visit(TrackElementPoly3 *poly3)
{
    Polynomial *newPoly = NULL;
    double l = 0.0;
    double h1 = 0.0;
    double dh1 = 0.0;

    if ((state_ == TrackMoveValidator::STATE_STARTPOINT)
        || (state_ == TrackMoveValidator::STATE_ENDPOINT))
    {
        QPointF endPoint;

        // Change start point //
        //
        if (state_ == TrackMoveValidator::STATE_STARTPOINT)
        {
            // Local to internal (Parameters are given in internal coordinates) //
            //
            QPointF deltaPos(poly3->getGlobalTransform().inverted().map(poly3->getGlobalPoint(poly3->getSStart()) + dPos_) - poly3->getPoint(poly3->getSStart()));
            endPoint = poly3->getPoint(poly3->getSEnd()) - deltaPos;
        }

        // Change end point //
        //
        else if (state_ == TrackMoveValidator::STATE_ENDPOINT)
        {
            endPoint = poly3->getGlobalTransform().inverted().map(poly3->getGlobalPoint(poly3->getSEnd()) + dPos_);
        }

        // Create Polynomial //
        //
        l = endPoint.x();
        double h0 = poly3->getA();
        double dh0 = poly3->getB();
        h1 = endPoint.y();
        dh1 = tan(poly3->getHeadingRad(poly3->getSEnd()));

        double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
        double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);

        newPoly = new Polynomial(h0, dh0, c, d);
    }

    if ((state_ == TrackMoveValidator::STATE_STARTHEADING)
        || (state_ == TrackMoveValidator::STATE_ENDHEADING))
    {
        QPointF endPoint;
        double heading = 0.0; // [rad]

        // Change start heading //
        //
        if (state_ == TrackMoveValidator::STATE_STARTHEADING)
        {
            // Global to internal (Parameters are given in internal coordinates) //
            //
            double deltaHeading = heading_ - poly3->getGlobalHeading(poly3->getSStart()); // heading_ is absolute

            QTransform trafo;
            trafo.rotate(deltaHeading);

            heading = poly3->getHeading(poly3->getSEnd()) - deltaHeading;
            endPoint = trafo.inverted().map(poly3->getPoint(poly3->getSEnd()));
        }

        // Change end heading //
        //
        else if (state_ == TrackMoveValidator::STATE_ENDHEADING)
        {
            // Global to internal (Parameters are given in internal coordinates) //
            //
            heading = heading_ - poly3->getGlobalHeading(poly3->getSStart());

            endPoint = poly3->getPoint(poly3->getSEnd());
        }

        // Check heading //
        //
        while (heading >= 360.0)
        {
            heading = heading - 360.0;
        }
        while (heading < 0.0)
        {
            heading = heading + 360.0;
        }
        if (heading >= 90.0 && heading <= 270.0)
        {
            isValid_ = false;
            return;
        }
        heading = heading * 2.0 * M_PI / 360.0;

        // Create Polynomial //
        //
        l = endPoint.x();
        double h0 = poly3->getA();
        double dh0 = poly3->getB();
        h1 = endPoint.y();
        dh1 = tan(heading);

        double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
        double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);

        newPoly = new Polynomial(h0, dh0, c, d);
    }

    // Must be longer than 1m //
    //
    if (l < 1.0) // TODO: hard coded = bad style
    {
        isValid_ = false;
        //		qDebug("d");
    }

    // No overshooting //
    //
    if (newPoly->ddf(0.0) > 0.0) // left turn
    {
        if (dh1 < -1.0) // max 45°
        {
            isValid_ = false;
            //			qDebug("a");
        }
    }

    if (newPoly->ddf(0.0) < 0.0) // right turn
    {
        if (dh1 > 1.0) // max 45°
        {
            isValid_ = false;
            //			qDebug("b");
        }
    }

    // No sharp corners //
    //
    if ((fabs(newPoly->k(0.0)) > 0.25)
        || (fabs(newPoly->k(l)) > 0.25))
    {
        isValid_ = false;
        //		qDebug("c");
    }

    delete newPoly;
}
