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

#ifndef RSYSTEMELEMENTFIDDLEYARD_HPP
#define RSYSTEMELEMENTFIDDLEYARD_HPP

#include "roadsystem.hpp"

#include <QMap>

// TODO OBSERVER, DATAELEMENT, PARENTING

//########################//
//                        //
// FiddleyardSource       //
//                        //
//########################//

class FiddleyardSource : public Acceptor
{

public:
    FiddleyardSource(const odrID &id, int lane, double startTime, double repeatTime, double velocity, double velocityDeviance);

    // <source> //
    //
    odrID getId() const
    {
        return id_;
    }
    int getLane() const
    {
        return lane_;
    }
    double getStartTime() const
    {
        return startTime_;
    }
    double getRepeatTime() const
    {
        return repeatTime_;
    }
    double getVelocity() const
    {
        return velocity_;
    }
    double getVelocityDeviance() const
    {
        return velocityDeviance_;
    }

    void setId(const odrID &id)
    {
        id_ = id;
    };

    // <source><vehicle> //
    //
    QMap<odrID, double> getVehicles() const
    {
        return vehicles_;
    }
    void addVehicle(const odrID &id, double numerator);

    // Prototype Pattern //
    //
    FiddleyardSource *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    FiddleyardSource()
    { /*not allowed*/
    }
    FiddleyardSource(const FiddleyardSource &); /* not allowed */
    FiddleyardSource &operator=(const FiddleyardSource &); /* not allowed */

private:
    odrID id_;
    int lane_;

    double startTime_;
    double repeatTime_;
    double velocity_;
    double velocityDeviance_;

    QMap<odrID, double> vehicles_;
};

//########################//
//                        //
// FiddleyardSink         //
//                        //
//########################//

class FiddleyardSink : public Acceptor
{

public:
    FiddleyardSink(const odrID &id, int lane);

    odrID getId() const
    {
        return id_;
    }
    int getLane() const
    {
        return lane_;
    }

    void setId(const odrID &id)
    {
        id_ = id;
    };

    // Prototype Pattern //
    //
    FiddleyardSink *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    FiddleyardSink()
    { /*not allowed*/
    }
    FiddleyardSink(const FiddleyardSink &); /* not allowed */
    FiddleyardSink &operator=(const FiddleyardSink &); /* not allowed */

private:
    odrID id_;
    int lane_;
};

//##########################//
//                          //
// RSystemElementFiddleyard //
//                          //
//##########################//

class RSystemElementFiddleyard : public RSystemElement
{
public:
    RSystemElementFiddleyard(const QString &name, const odrID &id, const QString &elementType, const odrID &elementId, const QString &contactPoint);
    virtual ~RSystemElementFiddleyard();

    // <source/sink> //
    //
    void addSource(FiddleyardSource *source);
    void addSink(FiddleyardSink *sink);
    QMap<odrID, FiddleyardSource *> getSources() const
    {
        return sources_;
    }
    QMap<odrID, FiddleyardSink *> getSinks() const
    {
        return sinks_;
    }

    // <link> //
    //
    QString getElementType() const
    {
        return elementType_;
    }
    const odrID &getElementId() const
    {
        return elementId_;
    }
    QString getContactPoint() const
    {
        return contactPoint_;
    }

    void setElementType(const QString &elementType);
    void setElementId(const odrID &elementId);
    void setContactPoint(const QString &contactPoint);

    // Prototype Pattern //
    //
    RSystemElementFiddleyard *getClone();

    void updateIds(const QMap<odrID, odrID> &roadIds);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForSources(Visitor *visitor);
    virtual void acceptForSinks(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);

private:
    RSystemElementFiddleyard(); /* not allowed */
    RSystemElementFiddleyard(const RSystemElementFiddleyard &); /* not allowed */
    RSystemElementFiddleyard &operator=(const RSystemElementFiddleyard &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // <link> //
    //
    QString elementType_;
    odrID elementId_;
    QString contactPoint_;

    // <source> //
    //
    QMap<odrID, FiddleyardSource *> sources_; // owned

    // <sink> //
    //
    QMap<odrID, FiddleyardSink *> sinks_; // owned
};

#endif // RSYSTEMELEMENTFIDDLEYARD_HPP
