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

#ifndef OBJECTOBJECT_HPP
#define OBJECTOBJECT_HPP

#include "roadsection.hpp"

class ObjectCorner
{
    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectCorner(double u, double v, double z, double height)
    {
        u_ = u;
        v_ = v;
        z_ = z;
        height_ = height;
    };
    virtual ~ObjectCorner()
    { /* does nothing */
    }

    double getU() const
    {
        return u_;
    }
    void setU(const double u)
    {
        u_ = u;
    }

    double getV() const
    {
        return v_;
    }
    void setV(const double v)
    {
        v_ = v;
    }

    double getZ() const
    {
        return z_;
    }
    void setZ(const double z)
    {
        z_ = z;
    }

    double getHeight() const
    {
        return height_;
    }
    void setHeight(const double height)
    {
        height_ = height;
    }


private:
    double u_;
    double v_;
    double z_;
    double height_;
};

class Object : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum ObjectChange
    {
        CEL_ParameterChange = 0x1
    };

    enum ObjectOrientation
    {
        POSITIVE_TRACK_DIRECTION,
        NEGATIVE_TRACK_DIRECTION
    };

    struct ObjectProperties
    {
        double t;
        ObjectOrientation orientation;
        double zOffset;
        QString type;
        double validLength;
        double length;
        double width;
        double radius;
        double height;
        double hdg;
        double pitch;
        double roll;
        bool pole;
    };

    struct ObjectRepeatRecord
    {
        double s;
        double length;
        double distance;
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Object(const QString &id, const QString &name, const QString &type, double s, double t, double zOffset,
                    double validLength, ObjectOrientation orientation, double length, double width, double radius, double height, double hdg,
                    double pitch, double roll, bool pole, double repeatS, double repeatLength, double repeatDistance, const QString &textureFile);
    virtual ~Object()
    { /* does nothing */
    }

    // Object //
    //
    QString getId() const
    {
        return id_;
    }
    void setId(const QString &id)
    {
        id_ = id;
    }

    QString getName() const
    {
        return name_;
    }
    void setName(const QString &name)
    {
        name_ = name;
	modelFile_ = name;
    }

    QString getModelFileName() const
    {
        return name_;
    }
    void setModelFileName(const QString &name)
    {
        name_ = name;
	modelFile_ = name;
    }

    QString getTextureFileName() const
    {
        return textureFile_;
    }
    void setTextureFileName(const QString &name)
    {
        textureFile_ = name;
    }

    QString getType() const
    {
        return objectProps_.type;
    }
    void setType(const QString &type)
    {
        objectProps_.type = type;
    }

    double getT() const
    {
        return objectProps_.t;
    }
    void setT(const double t)
    {
        objectProps_.t = t;
    }

    double getzOffset() const
    {
        return objectProps_.zOffset;
    }
    void setzOffset(const double zOffset)
    {
        objectProps_.zOffset = zOffset;
    }

    double getValidLength() const
    {
        return objectProps_.validLength;
    }
    void setValidLength(const double validLength)
    {
        objectProps_.validLength = validLength;
    }

    ObjectOrientation getOrientation() const
    {
        return objectProps_.orientation;
    }
    void setOrientation(const ObjectOrientation orientation)
    {
        objectProps_.orientation = orientation;
    }

    double getLength() const
    {
        return objectProps_.length;
    }
    void setLength(const double length)
    {
        objectProps_.length = length;
    }

    double getWidth() const
    {
        return objectProps_.width;
    }
    void setWidth(const double width)
    {
        objectProps_.width = width;
    }

    double getRadius() const
    {
        return objectProps_.radius;
    }
    void setRadius(const double radius)
    {
        objectProps_.radius = radius;
    }

    double getHeight() const
    {
        return objectProps_.height;
    }
    void setHeight(const double height)
    {
        objectProps_.height = height;
    }

    double getHeading() const
    {
        return objectProps_.hdg;
    }
    void setHeiading(const double hdg)
    {
        objectProps_.hdg = hdg;
    }

    double getPitch() const
    {
        return objectProps_.pitch;
    }
    void setPitch(const double pitch)
    {
        objectProps_.pitch = pitch;
    }

    double getRoll() const
    {
        return objectProps_.roll;
    }
    void setRoll(const double roll)
    {
        objectProps_.roll = roll;
    }

    bool getPole() const
    {
        return objectProps_.pole;
    }
    void setPole(const bool pole)
    {
        objectProps_.pole = pole;
    }

    double getRepeatS() const
    {
        return objectRepeat_.s;
    }
    void setRepeatS(const double s)
    {
        objectRepeat_.s = s;
    }

    double getRepeatLength() const
    {
        return objectRepeat_.length;
    }
    void setRepeatLength(const double length)
    {
        objectRepeat_.length = length;
    }

    double getRepeatDistance() const
    {
        return objectRepeat_.distance;
    }
    void setRepeatDistance(const double distance)
    {
        objectRepeat_.distance = distance;
    }

    ObjectProperties getProperties() const
    {
        return objectProps_;
    }
    void setProperties(const ObjectProperties objectProps)
    {
        objectProps_ = objectProps;
    }

    ObjectRepeatRecord getRepeatProperties() const
    {
        return objectRepeat_;
    }
    void setRepeatProperties(const ObjectRepeatRecord objectRepeatProps)
    {
        objectRepeat_ = objectRepeatProps;
    }


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getObjectChanges() const
    {
        return objectChanges_;
    }
    void addObjectChanges(int changes);

    virtual double getSEnd() const;

    // Prototype Pattern //
    //
    Object *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    Object(); /* not allowed */
    Object(const Object &); /* not allowed */
    Object &operator=(const Object &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Object //
    //
    // Mandatory
    QString id_;
    QString name_;
    QString modelFile_;
    QString textureFile_;

    ObjectProperties objectProps_;
    ObjectRepeatRecord objectRepeat_;


    // Change flags //
    //
    int objectChanges_;
};

#endif // OBJECTOBJECT_HPP
