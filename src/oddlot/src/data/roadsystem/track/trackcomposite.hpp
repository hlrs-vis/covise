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

#ifndef TRACKCOMPOSITE_HPP
#define TRACKCOMPOSITE_HPP

#include "trackcomponent.hpp"

#include <QMap>

class TrackComposite : public TrackComponent
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackComposite();
    virtual ~TrackComposite();

    // Composite Pattern //
    //
    virtual bool addTrackComponent(TrackComponent *trackComponent);
    virtual bool delTrackComponent(TrackComponent *trackComponent);
    virtual int getNChildren() const
    {
        return trackComponents_.size();
    }
    virtual TrackComponent *getChild(double s) const;
    virtual QMap<double, TrackComponent *> getChildTrackComponents() const
    {
        return trackComponents_;
    }

    // Track Component //
    //
    virtual double getSStart() const;
    virtual double getSEnd() const;
    virtual double getLength() const
    {
        return cachedLength_;
    }

    virtual double getCurvature(double s);

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

    virtual void setSStart(double s);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);

protected:
private:
    //TrackComposite(); /* not allowed */
    TrackComposite(const TrackComposite &); /* not allowed */
    TrackComposite &operator=(const TrackComposite &); /* not allowed */

    // Cached Values //
    //
    double updateLength();

    //################//
    // PROPERTIES     //
    //################//

protected:
    // Composite Pattern //
    //
    QMap<double, TrackComponent *> trackComponents_; // owned

private:
    // Cached Values //
    //
    double cachedLength_;
};

#endif // TRACKCOMPOSITE_HPP
