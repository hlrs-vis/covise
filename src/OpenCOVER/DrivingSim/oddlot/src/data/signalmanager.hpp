/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.05.2010
**
**************************************************************************/

#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <QObject>

#include <QString>
#include <QIcon>
#include <QMultiMap>

class ObjectCorner;

class SignalContainer
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalContainer(const QString &name, const QIcon &icon, int type, const QString &typeSubclass, int subType, double value, double distance, double height)
        : signalName_(name)
        , signalIcon_(icon)
        , signalType_(type)
        , signalTypeSubclass_(typeSubclass)
        , signalSubType_(subType)
        , signalValue_(value)
        , signalDistance_(distance)
        , signalHeight_(height)
    {
        /* does nothing */
    }

    virtual ~SignalContainer()
    { /* does nothing */
    }

    QString getSignalName() const
    {
        return signalName_;
    }
    QIcon getSignalIcon() const
    {
        return signalIcon_;
    }
    int getSignalType() const
    {
        return signalType_;
    }
    QString getSignalTypeSubclass() const
    {
        return signalTypeSubclass_;
    }
    int getSignalSubType() const
    {
        return signalSubType_;
    }
    double getSignalValue() const
    {
        return signalValue_;
    }
    double getSignalDistance() const
    {
        return signalDistance_;
    }
    double getSignalHeight() const
    {
        return signalHeight_;
    }

protected:
private:
    SignalContainer(); /* not allowed */
    SignalContainer(const SignalContainer &); /* not allowed */
    SignalContainer &operator=(const SignalContainer &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    QString signalName_;
    QIcon signalIcon_;
    int signalType_;
    QString signalTypeSubclass_;
    int signalSubType_;
    double signalValue_;
    double signalDistance_;
    double signalHeight_;
};

class ObjectContainer
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectContainer(const QString &name, const QIcon &icon, const QString &type, double length, double width, double radius, double height, double distance, double heading, double repeatDistance, const QList<ObjectCorner *> &corners)
        : objectName_(name)
        , objectIcon_(icon)
        , objectType_(type)
        , objectLength_(length)
        , objectWidth_(width)
        , objectRadius_(radius)
        , objectHeight_(height)
        , objectDistance_(distance)
        , objectHeading_(heading)
        , objectRepeatDistance_(repeatDistance)
        , objectCorners_(corners)
    {
        /* does nothing */
    }

    virtual ~ObjectContainer()
    { /* does nothing */
    }

    QString getObjectName() const
    {
        return objectName_;
    }
    QIcon getObjectIcon() const
    {
        return objectIcon_;
    }
    QString getObjectType() const
    {
        return objectType_;
    }
    double getObjectLength() const
    {
        return objectLength_;
    }
    double getObjectWidth() const
    {
        return objectWidth_;
    }
    double getObjectHeight() const
    {
        return objectHeight_;
    }
    double getObjectRadius() const
    {
        return objectRadius_;
    }
    double getObjectDistance() const
    {
        return objectDistance_;
    }
    double getObjectHeading() const
    {
        return objectHeading_;
    }
    double getObjectRepeatDistance() const
    {
        return objectRepeatDistance_;
    };
    QList<ObjectCorner *> getObjectCorners() const
    {
        return objectCorners_;
    };

protected:
private:
    ObjectContainer(); /* not allowed */
    ObjectContainer(const ObjectContainer &); /* not allowed */
    ObjectContainer &operator=(const ObjectContainer &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    QString objectName_;
    QIcon objectIcon_;
    QString objectType_;
    double objectLength_;
    double objectWidth_;
    double objectRadius_;
    double objectHeading_;
    double objectDistance_;
    double objectHeight_;
    double objectRepeatDistance_;
    QList<ObjectCorner *> objectCorners_;
};

class SignalManager : public QObject
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalManager(QObject *parent);
    virtual ~SignalManager();

    // User Prototypes //
    //
    bool loadSignals(const QString &fileName);

    void addSignal(const QString &country, const QString &name, const QIcon &icon, int type, const QString &typeSubclass, int subType, double value, double distance, double height);
    void addObject(const QString &country, const QString &name, const QIcon &icon, const QString &type, double length, double width, double radius, double height, double distance, double heading, double repeatDistance, const QList<ObjectCorner *> &corners);
    QList<SignalContainer *> getSignals(QString country) const
    {
        return signals_.values(country);
    };
    QList<ObjectContainer *> getObjects(QString country) const
    {
        return objects_.values(country);
    };

protected:
private:
    SignalManager(); /* not allowed */
    SignalManager(const SignalManager &); /* not allowed */
    SignalManager &operator=(const SignalManager &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // User Signals //
    //
    QMultiMap<QString, SignalContainer *> signals_;

    // User Objects //
    //
    QMultiMap<QString, ObjectContainer *> objects_;
};

#endif // SIGNAL_HPP
