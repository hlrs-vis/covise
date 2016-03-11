/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   1/18/2010
**
**************************************************************************/

#ifndef TRACKELEMENT_HPP
#define TRACKELEMENT_HPP

#include "trackcomponent.hpp"

class QPointF;

class TrackElement : public TrackComponent
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackElement(double x, double y, double angleDegrees, double s, double length);
    virtual ~TrackElement();

    // Track Element //
    //
    virtual void setLength(double length);
    virtual void setSStart(double s);

    // Track Component //
    //
    virtual double getSStart() const
    {
        return s_;
    }
    virtual double getSEnd() const
    {
        return s_ + length_;
    }
    virtual double getLength() const
    {
        return length_;
    }

    virtual int getStartRotDOF() const
    {
        return 0;
    }
    virtual int getEndRotDOF() const
    {
        return 0;
    }

    virtual void setLocalStartPoint(const QPointF & /*startPoint*/)
    { /* does nothing */
    }
    virtual void setLocalEndPoint(const QPointF & /*endPoint*/)
    { /* does nothing */
    }
    virtual void setLocalStartHeading(double /*startHeading*/)
    { /* does nothing */
    }
    virtual void setLocalEndHeading(double /*endHeading*/)
    { /* does nothing */
    }

    virtual void setLocalPointAndHeading(const QPointF & /*point*/, double /*heading*/, bool /*isStart*/)
    { /* does nothing */
    }

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor) = 0;

protected:
private:
    TrackElement(); /* not allowed */
    TrackElement(const TrackElement &); /* not allowed */
    TrackElement &operator=(const TrackElement &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Track Element //
    //
    double s_; // start position on parent road
    double length_; // length of this element
};

#endif // TRACKELEMENT_HPP
