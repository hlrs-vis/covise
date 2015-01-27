/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/21/2010
**
**************************************************************************/

#ifndef TRACKELEMENTPOLY3_HPP
#define TRACKELEMENTPOLY3_HPP

#include "trackelement.hpp"
#include "src/util/math/polynomial.hpp"

class TrackElementPoly3 : public TrackElement, public Polynomial
{

    //################//
    // STATIC         //
    //################//

public:
    enum TrackElementPoly3Change
    {
        CTP_ParameterChange = 0x1,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackElementPoly3(double x, double y, double angleDegrees, double s, double length, double a, double b, double c, double d);
    explicit TrackElementPoly3(double x, double y, double angleDegrees, double s, double length, const Polynomial &polynomial);
    virtual ~TrackElementPoly3();

    // Track Component //
    //
    virtual double getCurvature(double s);

    virtual QPointF getPoint(double s, double d = 0.0);
    virtual double getHeading(double s);
    virtual double getHeadingRad(double s);

    virtual QPointF getLocalPoint(double s, double d = 0.0);
    virtual double getLocalHeading(double s);
    virtual double getLocalHeadingRad(double s);
    virtual QVector2D getLocalTangent(double s);
    virtual QVector2D getLocalNormal(double s);

    virtual int getStartPosDOF() const
    {
        return 2;
    }
    virtual int getEndPosDOF() const
    {
        return 2;
    }
    virtual int getStartRotDOF() const
    {
        return 1;
    }
    virtual int getEndRotDOF() const
    {
        return 1;
    }

    virtual void setLocalStartPoint(const QPointF &startPoint);
    virtual void setLocalEndPoint(const QPointF &endPoint);
    virtual void setLocalStartHeading(double startHeading);
    virtual void setLocalEndHeading(double endHeading);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackElementPoly3Changes() const
    {
        return trackElementPoly3Changes_;
    }
    void addTrackElementPoly3Changes(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const;
    TrackElementPoly3 *getClonedPoly3() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    TrackElementPoly3(); /* not allowed: use clone() instead */
    TrackElementPoly3(const TrackElementPoly3 &); /* not allowed: use clone() instead */
    TrackElementPoly3 &operator=(const TrackElementPoly3 &); /* not allowed: use clone() instead */

    double getT(double s);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Observer Pattern //
    //
    int trackElementPoly3Changes_;
};

#endif // TRACKELEMENTPOLY3_HPP
