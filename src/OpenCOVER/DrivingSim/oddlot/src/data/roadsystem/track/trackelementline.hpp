/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   1/22/2010
**
**************************************************************************/

#ifndef TRACKELEMENTLINE_HPP
#define TRACKELEMENTLINE_HPP

#include "trackelement.hpp"

class TrackElementLine : public TrackElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum TrackElementLineChange
    {
        //		CTL_ Change				= 0x1,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackElementLine(double x, double y, double angleDegrees, double s, double length);
    virtual ~TrackElementLine();

    // Track Component //
    //
    virtual int getStartPosDOF() const
    {
        return 1;
    }
    virtual int getEndPosDOF() const
    {
        return 1;
    }

    virtual double getCurvature(double s);

    virtual QPointF getPoint(double s, double d = 0.0);
    virtual double getHeading(double s);
    virtual double getHeadingRad(double s);

    virtual QPointF getLocalPoint(double s, double d = 0.0);
    virtual double getLocalHeading(double s);
    virtual double getLocalHeadingRad(double s);
    virtual QVector2D getLocalTangent(double s);
    virtual QVector2D getLocalNormal(double s);

    virtual void setLocalStartPoint(const QPointF &startPoint);
    virtual void setLocalEndPoint(const QPointF &endPoint);
    virtual void setLocalStartHeading(double startHeadingDegrees);
    virtual void setLocalEndHeading(double endHeadingDegrees);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackElementLineChanges() const
    {
        return trackElementLineChanges_;
    }
    void addTrackElementLineChanges(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const;
    TrackElementLine *getClonedLine() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    TrackElementLine(); /* not allowed: use clone() instead */
    TrackElementLine(const TrackElementLine &); /* not allowed: use clone() instead */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Observer Pattern //
    //
    int trackElementLineChanges_;
};

#endif // TRACKELEMENTLINE_HPP
