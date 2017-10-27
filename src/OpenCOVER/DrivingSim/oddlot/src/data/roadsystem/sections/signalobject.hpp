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

#ifndef SIGNALOBJECT_HPP
#define SIGNALOBJECT_HPP

#include "roadsection.hpp"

class Signal : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum OrientationType
    {
        POSITIVE_TRACK_DIRECTION = 1,
        NEGATIVE_TRACK_DIRECTION = 2,
        BOTH_DIRECTIONS = 0
    };
    enum SignalChange
    {
        CEL_ParameterChange = 0x1,
        CEL_TypeChange = 0x2
    };

    struct SignalProperties
    {
        double t;
        bool dynamic;
        OrientationType orientation;
        double zOffset;
        QString country;
		QString type;
		QString subtype;
        double value;
        double hOffset;
        double pitch;
        double roll;
        bool pole;
		QString unit;
		QString text;
		double width;
		double height;
    };

    struct Validity
    {
        int fromLane;
        int toLane;
    };

    struct SignalUserData
    {
        double crossProb;
        double resetTime;
        QString typeSubclass;
        int size;
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Signal(const QString &id, const QString &name, double s, double t, bool dynamic, OrientationType orientation, double zOffset, QString country, const QString &type, const QString typeSubclass, const QString &subtype, double value, double hOffset, double pitch, double roll, QString unit, QString text, double width, double height, bool pole, int size, int validFromLane, int validToLane, double probability = 0.75, double resetTime = 20.0);
    explicit Signal(const QString &id, const QString &name, double s, SignalProperties &signalProps, Validity &validity, SignalUserData &userData);
    virtual ~Signal()
    { /* does nothing */
    }

    // Signal //
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
    }

    double getT() const
    {
        return signalProps_.t;
    }
    void setT(const double t)
    {
        signalProps_.t = t;
    }

    bool getDynamic() const
    {
        return signalProps_.dynamic;
    }
    void setDynamic(const bool dynamic)
    {
        signalProps_.dynamic = dynamic;
    }

    OrientationType getOrientation() const
    {
        return signalProps_.orientation;
    }
    void setOrientation(const OrientationType orientation)
    {
        signalProps_.orientation = orientation;
    }

    double getZOffset() const
    {
        return signalProps_.zOffset;
    }
    void setZOffset(const double zOffset)
    {
        signalProps_.zOffset = zOffset;
    }

    QString getCountry() const
    {
        return signalProps_.country;
    }
    void setCountry(const QString &country)
    {
        signalProps_.country = country;
    }

    QString getType() const
    {
        return signalProps_.type;
    }
    void setType(const QString &type)
    {
        signalProps_.type = type;
    }

    QString getTypeSubclass() const
    {
        return signalUserData_.typeSubclass;
    }
    void setTypeSubclass(const QString &typeSubclass)
    {
        signalUserData_.typeSubclass = typeSubclass;
    }

	QString getSubtype() const
    {
        return signalProps_.subtype;
    }
    void setSubtype(const QString &subtype)
    {
        signalProps_.subtype = subtype;
    }

    double getValue() const
    {
        return signalProps_.value;
    }
    void setValue(const double value)
    {
        signalProps_.value = value;
    }

    double getHeading() const
    {
        return signalProps_.hOffset;
    }
    void setHeading(const double hOffset)
    {
        signalProps_.hOffset = hOffset;
    }

    double getPitch() const
    {
        return signalProps_.pitch;
    }
    void setPitch(const double pitch)
    {
        signalProps_.pitch = pitch;
    }

    double getRoll() const
    {
        return signalProps_.roll;
    }
    void setRoll(const double roll)
    {
        signalProps_.roll = roll;
    }

    bool getPole() const
    {
        return signalProps_.pole;
    }
    void setPole(const bool pole)
    {
        signalProps_.pole = pole;
    }

    int getSize() const
    {
        return signalUserData_.size;
    }
    void setSize(int size)
    {
        signalUserData_.size = size;
    }

    SignalProperties getProperties() const
    {
        return signalProps_;
    }
    void setProperties(const SignalProperties & signalProps)
    {
        signalProps_ = signalProps;
    }

    double getSEnd() const
    {
        return getSStart();
    };
    double getLength() const
    {
        return 0;
    };

    int getValidFromLane() const
    {
        return validity_.fromLane;
    }
    void setValidFromLane(const int validFromLane)
    {
        validity_.fromLane = validFromLane;
    }

    int getValidToLane() const
    {
        return validity_.toLane;
    }
    void setValidToLane(const int validToLane)
    {
        validity_.toLane = validToLane;
    }

    Validity getValidity() const
    {
        return validity_;
    }
    void setValidity(Validity & validLanes)
    {
        validity_ = validLanes;
    }

    double getCrossingProbability() const
    {
        return signalUserData_.crossProb;
    }
    void setCrossingProbability(const double probability)
    {
        signalUserData_.crossProb = probability;
    }

    double getResetTime() const
    {
        return signalUserData_.resetTime;
    }
    void setResetTime(const double resetTime)
    {
        signalUserData_.resetTime = resetTime;
    }

	double getWidth() const
	{
		return signalProps_.width;
	}
	void setWidth(const double width)
	{
		signalProps_.width = width;
	}

	double getHeight() const
	{
		return signalProps_.height;
	}
	void setHeight(const double height)
	{
		signalProps_.height = height;
	}

	QString getUnit() const
	{
		return signalProps_.unit;
	}
	void setUnit(const QString unit)
	{
		signalProps_.unit = unit;
	}
	QString getText() const
	{
		return signalProps_.text;
	}
	void setText(const QString text)
	{
		signalProps_.text = text;
	}

    SignalUserData getSignalUserData() const
    {
        return signalUserData_;
    }
    void setSignalUserData(SignalUserData & userData)
    {
        signalUserData_ = userData;
    }


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getSignalChanges() const
    {
        return signalChanges_;
    }
    void addSignalChanges(int changes);

    // Prototype Pattern //
    //
    Signal *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    Signal(); /* not allowed */
    Signal(const Signal &); /* not allowed */
    Signal &operator=(const Signal &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Signal //
    //
    // Mandatory
    QString id_;
    QString name_;
    SignalProperties signalProps_;
    Validity validity_;
    SignalUserData signalUserData_;

    // Change flags //
    //
    int signalChanges_;
};

#endif // SIGNALOBJECT_HPP
