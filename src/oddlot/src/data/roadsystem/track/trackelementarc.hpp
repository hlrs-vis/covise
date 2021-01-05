/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.02.2010
**
**************************************************************************/

#ifndef TRACKELEMENTARC_HPP
#define TRACKELEMENTARC_HPP

#include "trackelement.hpp"

class TrackElementArc : public TrackElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum TrackElementArcChange
    {
        CTA_CurvChange = 0x1,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackElementArc(double x, double y, double angleDegrees, double s, double length, double curvature);
    virtual ~TrackElementArc();

    // Arc //
    //
    void setCurvature(double curvature);
    //	QPointF					getRadiusCenter(double s);

    // Track Component //
    //
    virtual int getStartPosDOF() const
    {
        return 0;
    }
    virtual int getEndPosDOF() const
    {
        return 0;
    }

    virtual double getCurvature(double s);
    virtual double getRadius(double s);

    virtual QPointF getPoint(double s, double d = 0.0);
    virtual double getHeading(double s);
    virtual double getHeadingRad(double s);

    virtual QPointF getLocalPoint(double s, double d = 0.0);
    virtual double getLocalHeading(double s);
    virtual double getLocalHeadingRad(double s);
    virtual QVector2D getLocalTangent(double s);
    virtual QVector2D getLocalNormal(double s);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackElementArcChanges() const
    {
        return trackElementArcChanges_;
    }
    void addTrackElementArcChanges(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const;
    TrackElementArc *getClonedArc() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    TrackElementArc(); /* not allowed: use clone() instead */
    TrackElementArc(const TrackElementArc &); /* not allowed: use clone() instead */
    TrackElementArc &operator=(const TrackElementArc &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Arc //
    //
    double curvature_;

    // Observer Pattern //
    //
    int trackElementArcChanges_;
};

#endif // TRACKELEMENTARC_HPP
