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

#include "rsystemelement.hpp"

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
    FiddleyardSource(QString &id, int lane, double startTime, double repeatTime, double velocity, double velocityDeviance);

    // <source> //
    //
    QString getId() const
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

    void setId(const QString &id)
    {
        id_ = id;
    };

    // <source><vehicle> //
    //
    QMap<QString, double> getVehicles() const
    {
        return vehicles_;
    }
    void addVehicle(const QString &id, double numerator);

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
    QString id_;
    int lane_;

    double startTime_;
    double repeatTime_;
    double velocity_;
    double velocityDeviance_;

    QMap<QString, double> vehicles_;
};

//########################//
//                        //
// FiddleyardSink         //
//                        //
//########################//

class FiddleyardSink : public Acceptor
{

public:
    FiddleyardSink(const QString &id, int lane);

    QString getId() const
    {
        return id_;
    }
    int getLane() const
    {
        return lane_;
    }

    void setId(const QString &id)
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
    QString id_;
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
    RSystemElementFiddleyard(const QString &name, const QString &id, const QString &elementType, const QString &elementId, const QString &contactPoint);
    virtual ~RSystemElementFiddleyard();

    // <source/sink> //
    //
    void addSource(FiddleyardSource *source);
    void addSink(FiddleyardSink *sink);
    QMap<QString, FiddleyardSource *> getSources() const
    {
        return sources_;
    }
    QMap<QString, FiddleyardSink *> getSinks() const
    {
        return sinks_;
    }

    // <link> //
    //
    QString getElementType() const
    {
        return elementType_;
    }
    QString getElementId() const
    {
        return elementId_;
    }
    QString getContactPoint() const
    {
        return contactPoint_;
    }

    void setElementType(const QString &elementType);
    void setElementId(const QString &elementId);
    void setContactPoint(const QString &contactPoint);

    // Prototype Pattern //
    //
    RSystemElementFiddleyard *getClone();

    void updateIds(const QMap<QString, QString> &roadIds);

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
    QString elementId_;
    QString contactPoint_;

    // <source> //
    //
    QMap<QString, FiddleyardSource *> sources_; // owned

    // <sink> //
    //
    QMap<QString, FiddleyardSink *> sinks_; // owned
};

#endif // RSYSTEMELEMENTFIDDLEYARD_HPP
