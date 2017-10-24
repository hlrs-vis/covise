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

class ParkingSpace;

class Object : public RoadSection
{
	friend ParkingSpace;

    //################//
    // STATIC         //
    //################//

public:
    enum ObjectChange
    {
        CEL_ParameterChange = 0x1,
		CEL_TypeChange = 0x2,
		CEL_ParkingSpaceChange = 0x4
    };

    enum ObjectOrientation
    {
        POSITIVE_TRACK_DIRECTION = 1,
        NEGATIVE_TRACK_DIRECTION = 2,
        BOTH_DIRECTIONS = 0
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
		double tStart;
		double tEnd;
		double widthStart;
		double widthEnd;
		double heightStart;
		double heightEnd;
		double zOffsetStart;
		double zOffsetEnd;
    };

    struct ObjectUserData
    {
        QString textureFile;
        QString modelFile;
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit Object(const QString &id, const QString &name, double s, ObjectProperties &objectProps, ObjectRepeatRecord &repeatRecord, const QString &textureFile);
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
        userData_.modelFile = name;
    }

    QString getModelFileName() const
    {
        return name_;
    }
    void setModelFileName(const QString &name)
    {
        name_ = name;
	    userData_.modelFile = name;
    }

    QString getTextureFileName() const
    {
        return userData_.textureFile;
    }
    void setTextureFileName(const QString &name)
    {
        userData_.textureFile = name;
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
    void setHeading(const double hdg)
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

	double getRepeatTStart() const
	{
		return objectRepeat_.tStart;
	}
	void setRepeatTStart(const double tStart)
	{
		objectRepeat_.tStart = tStart;
	}

	double getRepeatTEnd() const
	{
		return objectRepeat_.tEnd;
	}
	void setRepeatTEnd(const double tEnd)
	{
		objectRepeat_.tEnd = tEnd;
	}

	double getRepeatWidthStart() const
	{
		return objectRepeat_.widthStart;
	}
	void setRepeatWidthStart(const double widthStart)
	{
		objectRepeat_.widthStart = widthStart;
	}

	double getRepeatWidthEnd() const
	{
		return objectRepeat_.widthEnd;
	}
	void setRepeatWidthEnd(const double widthEnd)
	{
		objectRepeat_.widthEnd = widthEnd;
	}

	double getRepeatHeightStart() const
	{
		return objectRepeat_.heightStart;
	}
	void setRepeatHeightStart(const double heightStart)
	{
		objectRepeat_.heightStart = heightStart;
	}

	double getRepeatHeightEnd() const
	{
		return objectRepeat_.heightEnd;
	}
	void setRepeatHeightEnd(const double heightEnd)
	{
		objectRepeat_.heightEnd = heightEnd;
	}

	double getRepeatZOffsetStart() const
	{
		return objectRepeat_.zOffsetStart;
	}
	void setRepeatZOffsetStart(const double zOffsetStart)
	{
		objectRepeat_.zOffsetStart = zOffsetStart;
	}

	double getRepeatZOffsetEnd() const
	{
		return objectRepeat_.zOffsetEnd;
	}
	void setRepeatZOffsetEnd(const double zOffsetEnd)
	{
		objectRepeat_.zOffsetEnd = zOffsetEnd;
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

	// Object is parking space //
	//
	ParkingSpace *getParkingSpace()
	{
		return parkingSpace_;
	}
	void setParkingSpace(ParkingSpace *parkingSpace);


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

    ObjectProperties objectProps_;
    ObjectRepeatRecord objectRepeat_;
    ObjectUserData userData_;

	ParkingSpace *parkingSpace_;


    // Change flags //
    //
    int objectChanges_;
};

#endif // OBJECTOBJECT_HPP
