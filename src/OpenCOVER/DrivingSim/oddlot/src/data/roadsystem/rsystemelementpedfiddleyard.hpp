/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#ifndef RSYSTEMELEMENTPEDFIDDLEYARD_HPP
#define RSYSTEMELEMENTPEDFIDDLEYARD_HPP

#include "rsystemelement.hpp"

#include <QMap>

// TODO OBSERVER, DATAELEMENT, PARENTING

//########################//
//                        //
// PedFiddleyardSource    //
//                        //
//########################//

class PedFiddleyardSource : public Acceptor
{

public:
    PedFiddleyardSource(const QString &id, int lane, double velocity);

    // <source> //
    // Mandatory attributes
    QString getId() const
    {
        return id_;
    }
    int getLane() const
    {
        return lane_;
    }
    double getVelocity() const
    {
        return velocity_;
    }

    // Optional attributes
    void setStartTime(double s)
    {
        startTime_ = s;
        startTimeSet_ = true;
    }
    double getStartTime() const
    {
        return startTime_;
    }
    bool hasStartTime() const
    {
        return startTimeSet_;
    }

    void setRepeatTime(double r)
    {
        repeatTime_ = r;
        repeatTimeSet_ = true;
    }
    double getRepeatTime() const
    {
        return repeatTime_;
    }
    bool hasRepeatTime() const
    {
        return repeatTimeSet_;
    }

    void setTimeDeviance(double t)
    {
        timeDeviance_ = t;
        timeDevianceSet_ = true;
    }
    double getTimeDeviance() const
    {
        return timeDeviance_;
    }
    bool hasTimeDeviance() const
    {
        return timeDevianceSet_;
    }

    void setDirection(int d)
    {
        direction_ = d;
        directionSet_ = true;
    }
    int getDirection() const
    {
        return direction_;
    }
    bool hasDirection() const
    {
        return directionSet_;
    }

    void setSOffset(double s)
    {
        sOffset_ = s;
        sOffsetSet_ = true;
    }
    double getSOffset() const
    {
        return sOffset_;
    }
    bool hasSOffset() const
    {
        return sOffsetSet_;
    }

    void setVOffset(double v)
    {
        vOffset_ = v;
        vOffsetSet_ = true;
    }
    double getVOffset() const
    {
        return vOffset_;
    }
    bool hasVOffset() const
    {
        return vOffsetSet_;
    }

    void setVelocityDeviance(double d)
    {
        velocityDeviance_ = d;
        velocityDevianceSet_ = true;
    }
    double getVelocityDeviance() const
    {
        return velocityDeviance_;
    }
    bool hasVelocityDeviance() const
    {
        return velocityDevianceSet_;
    }

    void setAcceleration(double a)
    {
        acceleration_ = a;
        accelerationSet_ = true;
    }
    double getAcceleration() const
    {
        return acceleration_;
    }
    bool hasAcceleration() const
    {
        return accelerationSet_;
    }

    void setAccelerationDeviance(double d)
    {
        accelerationDeviance_ = d;
        accelerationDevianceSet_ = true;
    }
    double getAccelerationDeviance() const
    {
        return accelerationDeviance_;
    }
    bool hasAccelerationDeviance() const
    {
        return accelerationDevianceSet_;
    }

    // <source><ped> //
    //
    QMap<QString, double> getPedestrians() const
    {
        return peds_;
    }
    void addPedestrian(const QString &id, double numerator);

    // Prototype Pattern //
    //
    PedFiddleyardSource *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    PedFiddleyardSource()
    { /*not allowed*/
    }
    PedFiddleyardSource(const PedFiddleyardSource &); /* not allowed */
    PedFiddleyardSource &operator=(const PedFiddleyardSource &); /* not allowed */

private:
    QString id_;
    int lane_;
    double velocity_;

    double startTime_;
    bool startTimeSet_;

    double repeatTime_;
    bool repeatTimeSet_;

    double timeDeviance_;
    bool timeDevianceSet_;

    int direction_;
    bool directionSet_;

    double sOffset_;
    bool sOffsetSet_;

    double vOffset_;
    bool vOffsetSet_;

    double velocityDeviance_;
    bool velocityDevianceSet_;

    double acceleration_;
    bool accelerationSet_;

    double accelerationDeviance_;
    bool accelerationDevianceSet_;

    QMap<QString, double> peds_;
};

//########################//
//                        //
// PedFiddleyardSink      //
//                        //
//########################//

class PedFiddleyardSink : public Acceptor
{

public:
    PedFiddleyardSink(const QString &id, int lane);

    // Mandatory attributes
    QString getId() const
    {
        return id_;
    }
    int getLane() const
    {
        return lane_;
    }

    // Optional attributes
    void setSinkProb(double p)
    {
        sinkProb_ = p;
        sinkProbSet_ = true;
    }
    double getSinkProb() const
    {
        return sinkProb_;
    }
    bool hasSinkProb() const
    {
        return sinkProbSet_;
    }

    void setDirection(int d)
    {
        direction_ = d;
        directionSet_ = true;
    }
    int getDirection() const
    {
        return direction_;
    }
    bool hasDirection() const
    {
        return directionSet_;
    }

    void setSOffset(double s)
    {
        sOffset_ = s;
        sOffsetSet_ = true;
    }
    double getSOffset() const
    {
        return sOffset_;
    }
    bool hasSOffset() const
    {
        return sOffsetSet_;
    }

    void setVOffset(double v)
    {
        vOffset_ = v;
        vOffsetSet_ = true;
    }
    double getVOffset() const
    {
        return vOffset_;
    }
    bool hasVOffset() const
    {
        return vOffsetSet_;
    }

    // Prototype Pattern //
    //
    PedFiddleyardSink *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    PedFiddleyardSink()
    { /*not allowed*/
    }
    PedFiddleyardSink(const PedFiddleyardSink &); /* not allowed */
    PedFiddleyardSink &operator=(const PedFiddleyardSink &); /* not allowed */

private:
    QString id_;
    int lane_;

    double sinkProb_;
    bool sinkProbSet_;
    int direction_;
    bool directionSet_;
    double sOffset_;
    bool sOffsetSet_;
    double vOffset_;
    bool vOffsetSet_;
};

//#############################//
//                             //
// RSystemElementPedFiddleyard //
//                             //
//#############################//

class RSystemElementPedFiddleyard : public RSystemElement
{
public:
    RSystemElementPedFiddleyard(const QString &id, const QString &name, const QString &roadId);
    virtual ~RSystemElementPedFiddleyard();

    // <source/sink> //
    //
    void addSource(PedFiddleyardSource *source);
    void addSink(PedFiddleyardSink *sink);
    QMap<QString, PedFiddleyardSource *> getSources() const
    {
        return sources_;
    }
    QMap<QString, PedFiddleyardSink *> getSinks() const
    {
        return sinks_;
    }

    QString getRoadId() const
    {
        return roadId_;
    }

    // Prototype Pattern //
    //
    RSystemElementPedFiddleyard *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForSources(Visitor *visitor);
    virtual void acceptForSinks(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);

private:
    RSystemElementPedFiddleyard(); /* not allowed */
    RSystemElementPedFiddleyard(const RSystemElementPedFiddleyard &); /* not allowed */
    RSystemElementPedFiddleyard &operator=(const RSystemElementPedFiddleyard &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // <source> //
    //
    QMap<QString, PedFiddleyardSource *> sources_; // owned

    // <sink> //
    //
    QMap<QString, PedFiddleyardSink *> sinks_; // owned

    QString roadId_; // owned
};

#endif // RSYSTEMELEMENTPEDFIDDLEYARD_HPP
