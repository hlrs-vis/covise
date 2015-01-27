/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#ifndef BRIDGEOBJECT_HPP
#define BRIDGEOBJECT_HPP

#include "roadsection.hpp"

class Bridge : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum BridgeChange
    {
        CEL_ParameterChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Bridge(const QString &id, const QString &file, const QString &name, int type, double s, double length);
    virtual ~Bridge()
    { /* does nothing */
    }

    // Bridge //
    //
    QString getId() const
    {
        return id_;
    }
    void setId(const QString &id)
    {
        id_ = id;
    }
    QString getNewId(const QString &name);

    QString getName() const
    {
        return name_;
    }
    void setName(const QString &name)
    {
        name_ = name;
    }

    QString getFileName() const
    {
        return fileName_;
    }
    void setFileName(const QString &name)
    {
        fileName_ = name;
    }

    int getType() const
    {
        return type_;
    }
    void setType(const int &type)
    {
        type_ = type;
    }

    double getLength() const
    {
        return length_;
    }
    void setLength(const double length)
    {
        length_ = length;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getBridgeChanges() const
    {
        return bridgeChanges_;
    }
    void addBridgeChanges(int changes);

    virtual double getSEnd() const;

    // Prototype Pattern //
    //
    Bridge *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    Bridge(); /* not allowed */
    Bridge(const Bridge &); /* not allowed */
    Bridge &operator=(const Bridge &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Bridge //
    //
    // Mandatory
    QString id_;
    QString fileName_;
    QString name_;
    int type_;

    double length_;

    // Change flags //
    //
    int bridgeChanges_;
};

#endif // OBJECTOBJECT_HPP
