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

#ifndef TRACKELEMENTSPIRAL_HPP
#define TRACKELEMENTSPIRAL_HPP

#include "trackelement.hpp"

class TrackElementSpiral : public TrackElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum TrackElementSpiralChange
    {
        CTS_CurvStartChange = 0x1,
        CTS_CurvEndChange = 0x2,
    };

    static double x(double la);
    static double y(double la);

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackElementSpiral(double x, double y, double angleDegrees, double s, double length, double curvStart, double curvEnd);
    virtual ~TrackElementSpiral();

    // Spiral //
    //
    virtual int getStartPosDOF() const
    {
        return 0;
    }
    virtual int getEndPosDOF() const
    {
        return 0;
    }

    double getA() const
    {
        return ax_;
    }
    double getAsign() const
    {
        return ax_ / ay_;
    }

    void setCurvStartAndLength(double curvStart, double length);
    void setCurvEndAndLength(double curvEnd, double length);

    // Track Component //
    //
    virtual QPointF getPoint(double s, double d = 0.0);
    virtual double getHeading(double s);
    virtual double getHeadingRad(double s);
    virtual QVector2D getTangent(double s);
    virtual QVector2D getNormal(double s);

    virtual QPointF getLocalPoint(double s, double d = 0.0);
    virtual double getLocalHeading(double s);
    virtual double getLocalHeadingRad(double s);
    virtual QVector2D getLocalTangent(double s);
    virtual QVector2D getLocalNormal(double s);

    virtual double getCurvature(double s);
    virtual double getRadius(double s);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackElementSpiralChanges() const
    {
        return trackElementSpiralChanges_;
    }
    void addTrackElementSpiralChanges(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const;
    TrackElementSpiral *getClonedSpiral() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    TrackElementSpiral(); /* not allowed: use getClone() instead */
    TrackElementSpiral(const TrackElementSpiral &); /* not allowed: use getClone() instead */

    void init();

    // Internal Functions //
    //
    QPointF getRadiusCenter(double s);

    QPointF clothoidApproximation(double l);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Spiral //
    //
    double curvStart_;
    double curvEnd_;
    double ax_; // cached!
    double ay_; // cached!

    double lStart_; // cached!
    double headingStart_; // cached!
    QTransform clothoidTrafo_;

    // Series Expansion //
    //
    static double f0;
    static double f1;
    static double f2;
    static double f3;
    static double f4;
    static double f5;
    static double f6;
    static double f7;
    static double f8;

    // Observer Pattern //
    //
    int trackElementSpiralChanges_;
};

#endif // TRACKELEMENTSPIRAL_HPP
