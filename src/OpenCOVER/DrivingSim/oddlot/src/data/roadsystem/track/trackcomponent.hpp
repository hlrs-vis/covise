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

#ifndef TRACKCOMPONENT_HPP
#define TRACKCOMPONENT_HPP

#include "src/data/dataelement.hpp"

#include <cstdlib>

// Qt //
//
#include <QTransform>
#include <QPointF>
class QVector2D;

/*! \brief The base class for all track elements and composites.
*/
class TrackComponent : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum DTrackType
    {
        DTT_NONE,
        DTT_LINE,
        DTT_ARC,
        DTT_SPIRAL,
        DTT_POLY3,
		DTT_CUBICCURVE,
        DTT_COMPOSITE,
        DTT_SPARCS,
        DTT_USER
    };

    enum TrackComponentChange
    {
        CTC_ParentChanged = 0x1,
        CTC_SChange = 0x2,
        CTC_LengthChange = 0x4,
        CTC_TransformChange = 0x8, // position and heading
        CTC_ShapeChange = 0x10, // e.g. spiral parameters or arc radius
        CTC_AddedChild = 0x20,
        CTC_DeletedChild = 0x40
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackComponent(double x, double y, double angleDegrees);
    virtual ~TrackComponent();

    // Track Component //
    //
    virtual double getSStart() const = 0;
    virtual double getSEnd() const = 0;
    virtual double getLength() const = 0;
    virtual void setLength(double length);

    virtual int getStartPosDOF() const = 0;
    virtual int getEndPosDOF() const = 0;
    virtual int getStartRotDOF() const = 0;
    virtual int getEndRotDOF() const = 0;

    virtual double getCurvature(double s) = 0;

    virtual QPointF getPoint(double s, double d = 0.0) = 0;
    virtual double getHeading(double s) = 0;
    virtual double getHeadingRad(double s) = 0;

    virtual QPointF getLocalPoint(double s, double d = 0.0) = 0;
    virtual double getLocalHeading(double s) = 0;
    virtual double getLocalHeadingRad(double s) = 0;
    virtual QVector2D getLocalTangent(double s) = 0;
    virtual QVector2D getLocalNormal(double s) = 0;

    QPointF getGlobalPoint(double s, double d = 0.0);
    double getGlobalHeading(double s);
    double getGlobalHeadingRad(double s);
    QVector2D getGlobalTangent(double s);
    QVector2D getGlobalNormal(double s);

    virtual const QTransform getLocalTransform();
    virtual const QTransform getGlobalTransform();

    virtual void setSStart(double s) = 0;

    void setGlobalTranslation(const QPointF &pos);
    void setGlobalRotation(double angleDegrees);
    

    void setLocalTransform(const QPointF &pos, double angleDegrees);
    void setLocalTransform(double x, double y, double angleDegrees);
    //void					setLocalTransform(const QTransform & matrix); // TODO somewhere in the far future

    void setLocalTranslation(const QPointF &pos);
    void setLocalTranslation(double x, double y);
    void setLocalRotation(double angleDegrees);

    void translateLocal(const QPointF &pos);
    void translateLocal(double x, double y);
    void rotateLocal(double angleDegrees);

    // TODO: these should be implemented in a util class (visitor?) to maximize encapsulation!
    //  See Effective C++ (3rd edition) Item 23.
    //
    // Convenience function to modify one point and keep the other.
    void setGlobalStartPoint(const QPointF &startPoint);
    void setGlobalEndPoint(const QPointF &endPoint);
    void setGlobalStartHeading(double startHeading);
    void setGlobalEndHeading(double endHeading);

    virtual void setLocalStartPoint(const QPointF &startPoint) = 0;
    virtual void setLocalEndPoint(const QPointF &endPoint) = 0;
    virtual void setLocalStartHeading(double startHeading) = 0;
    virtual void setLocalEndHeading(double endHeading) = 0;

    // Convenience function to modify one end's location and rotation.
    void setGlobalPointAndHeading(const QPointF &point, double heading, bool isStart);

    virtual void setLocalPointAndHeading(const QPointF &point, double heading, bool isStart) = 0;
    //
    // End TODO.

    // TrackType //
    //
    TrackComponent::DTrackType getTrackType() const
    {
        return trackType_;
    }

    // Parents //
    //
    TrackComponent *getParentComponent() const;
    void setParentComponent(TrackComponent *parentComponent);

    RSystemElementRoad *getParentRoad() const;
    void setParentRoad(RSystemElementRoad *parentRoad);

    // Composite Pattern //
    //
    // most standard implementations do nothing - overwritten by composites
    //	virtual bool			add(TrackComponent *)		{ return false; }	// returns false on failing
    //	virtual bool			remove(TrackComponent *)	{ return false; }
    virtual int getNChildren()
    {
        return 0;
    } // returns number of children
    virtual TrackComponent *getChild(double /*s*/)
    {
        return NULL;
    }
    virtual QMap<double, TrackComponent *> getChildTrackComponents() const
    {
        return QMap<double, TrackComponent *>();
    }

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackComponentChanges() const
    {
        return trackComponentChanges_;
    }
    void addTrackComponentChanges(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const = 0; // implemented by subclasses

protected:
    // Convenience functions for child classes //
    //
    const QPointF &pos() const
    {
        return pos_;
    }
    double heading() const
    {
        return heading_;
    }

    void setTrackType(TrackComponent::DTrackType type);

private:
    TrackComponent(); /* not allowed */
    TrackComponent(const TrackComponent &); /* not allowed */
    TrackComponent &operator=(const TrackComponent &); /* not allowed */

    //################//
    // FUNCTIONS      //
    //################//

private:
    // TrackType //
    //
    TrackComponent::DTrackType trackType_;

    // Track Component //
    //
    QPointF pos_;
    double heading_; // DEGREES! (OpenDRIVE uses radians)

    // Parents //
    //
    RSystemElementRoad *parentRoad_; // linked
    TrackComponent *parentComponent_; // linked

    // Observer Pattern //
    //
    int trackComponentChanges_;
};

#endif // TRACKCOMPONENT_HPP
