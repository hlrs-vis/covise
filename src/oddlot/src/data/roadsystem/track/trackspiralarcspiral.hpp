/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.04.2010
**
**************************************************************************/

#ifndef TRACKSPIRALARCSPIRAL_HPP
#define TRACKSPIRALARCSPIRAL_HPP

#include "trackcomposite.hpp"

class TrackSpiralArcSpiral;

class SpArcSParameters
{
public:
    friend class TrackSpiralArcSpiral;

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SpArcSParameters(const QPointF &pEnd, double headingEnd, double factor);

    // Validation //
    //
    void init();
    bool isValid();

    // Parameters //
    //
    QPointF getEndPoint() const
    {
        return pEnd_;
    }
    void setEndPoint(const QPointF &endPoint);

    double getEndHeadingRad() const
    {
        return headingEnd_;
    }
    void setEndHeadingRad(double endHeadingRad);
    void setEndHeadingDeg(double endHeadingDeg);

    double getFactor() const
    {
        return factor_;
    }
    void setFactor(double factor);

    double getAngle() const
    {
        return angle_;
    }

private:
    SpArcSParameters(); /* not allowed */
    SpArcSParameters(const SpArcSParameters &); /* not allowed */

    bool checkValidity();

    double calcTau0u(bool &success);

    double f(double tau);
    double df(double tau);
    double q(double tau, double tau1);
    double dq(double tau);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Parameters //
    //
    QPointF pEnd_;
    double headingEnd_;
    double factor_;

    int isValid_;

    // Cache //
    //
    double angle_;
    double g_;
    double h_;
    double k_;

    double tau0_;
    double tau1_;
};

class TrackSpiralArcSpiral : public TrackComposite
{

    //################//
    // STATIC         //
    //################//

public:
    enum TrackSpArcSChange
    {
        CTV_ParameterChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackSpiralArcSpiral(TrackElementSpiral *inSpiral, TrackElementArc *arc, TrackElementSpiral *outSpiral);
    explicit TrackSpiralArcSpiral(const QPointF &startPos, const QPointF &endPos, double startHeadingDeg, double endHeadingDeg, double factor);
    virtual ~TrackSpiralArcSpiral();

    // SpArcS // symmetric
    //
    double getFactor() const
    {
        return pa_->factor_;
    }
    void setFactor(double factor);

    double calcFactor() const;

    TrackElementSpiral *getInSpiral() const
    {
        return inSpiral_;
    }
    TrackElementArc *getArc() const
    {
        return arc_;
    }
    TrackElementSpiral *getOutSpiral() const
    {
        return outSpiral_;
    }

    SpArcSParameters *getClonedParameters() const;
    bool validParameters() const
    {
        return validParameters_;
    }

    double getInTangentLength() const
    {
        return pa_->g_;
    }
    double getOutTangentLength() const
    {
        return pa_->h_;
    }

    // Track Component //
    //
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

    virtual void setLocalPointAndHeading(const QPointF &point, double heading, bool isStart);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackSpArcSChanges() const
    {
        return trackSpArcSChanges_;
    }
    void addTrackSpArcSChanges(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const;
    TrackSpiralArcSpiral *getClonedSpArcS() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    TrackSpiralArcSpiral(); /* not allowed */
    TrackSpiralArcSpiral(const TrackSpiralArcSpiral &); /* not allowed */

    void applyParameters();

    void setEndPoint(const QPointF &endPoint);
    void setEndHeadingDeg(double endHeading);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Tracks //
    //
    TrackElementSpiral *inSpiral_; // convenience (owned by base class)
    TrackElementArc *arc_; // convenience (owned by base class)
    TrackElementSpiral *outSpiral_; // convenience (owned by base class)

    // Observer Pattern //
    //
    int trackSpArcSChanges_;

    // SpArcS //
    //
    SpArcSParameters *pa_; // owned
    bool validParameters_;
};

#endif // TRACKSPIRALARCSPIRAL_HPP
