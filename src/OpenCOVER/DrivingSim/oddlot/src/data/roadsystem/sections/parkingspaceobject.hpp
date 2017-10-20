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

#ifndef PARKINGSPACEOBJECT_HPP
#define PARKINGSPACEOBJECT_HPP

#include "src/data/roadsystem/sections/laneroadmark.hpp"

class Object;
class ParkingSpace;


class ParkingSpaceMarking: public DataElement
{
public:
	// Observer Pattern //
	//
	enum ParkingSpaceMarkingChange
	{
		CPSM_ParentChanged = 0x1,
		CPSM_SideChanged = 0x2,
		CPSM_TypeChanged = 0x4,
		CPSM_WidthChanged = 0x8,
		CPSM_ColorChanged = 0x10
	};

	enum ParkingSpaceMarkingSide
	{
		PSM_NONE,
		PSM_FRONT,
		PSM_REAR,
		PSM_LEFT,
		PSM_RIGHT
	};

	static ParkingSpaceMarking::ParkingSpaceMarkingSide parseParkingSpaceMarkingSide(const QString &side);
	static QString parseParkingSpaceMarkingSideBack(ParkingSpaceMarking::ParkingSpaceMarkingSide side);

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit ParkingSpaceMarking(ParkingSpaceMarkingSide side, LaneRoadMark::RoadMarkType type, double width, LaneRoadMark::RoadMarkColor color);

    virtual ~ParkingSpaceMarking()
    { /* does nothing */
    }

	ParkingSpace *getParentParkingSpace()
	{
		return parentParkingSpace_;
	}
	void setParentParkingSpace(ParkingSpace *parkingSpace);

	ParkingSpaceMarkingSide getSide()
	{
		return side_;
	}
	void setSide(ParkingSpaceMarkingSide side);

	LaneRoadMark::RoadMarkType getType()
	{
		return type_;
	}
	void setType(LaneRoadMark::RoadMarkType type);

	double getWidth()
	{
		return width_;
	}
	void setWidth(double width);

	LaneRoadMark::RoadMarkColor getColor()
	{
		return color_;
	}
	void setColor(LaneRoadMark::RoadMarkColor color);

	// Observer Pattern //
	//
	virtual void notificationDone();
	int getParkingSpaceMarkingChanges() const
	{
		return markingChanges_;
	}
	void addParkingSpaceMarkingChanges(int changes);

	// Prototype Pattern //
	//
	ParkingSpaceMarking *getClone();

	// Visitor Pattern //
	//
	virtual void accept(Visitor *visitor);


private:
	ParkingSpace *parentParkingSpace_;

	ParkingSpaceMarkingSide side_;
	LaneRoadMark::RoadMarkType type_;
	double width_;
	LaneRoadMark::RoadMarkColor color_;

	int markingChanges_;
};

class ParkingSpace : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
	// Observer Pattern //
	//
	enum ParkingSpaceChange
	{
		CPS_ParentChanged = 0x1,
		CPS_AccessChanged = 0x2,
		CPS_RestrictionsChanged = 0x4,
		CPS_MarkingsChanged = 0x10
	};

	enum ParkingSpaceAccess
	{
		PS_NONE,
		PS_ALL,
		PS_CAR,
		PS_WOMEN,
	    PS_HANDICAPPED,
		PS_BUS,
		PS_TRUCK,
		PS_ELECTRIC,
		PS_RESIDENTS
    };

	static ParkingSpace::ParkingSpaceAccess parseParkingSpaceAccess(const QString &access);
	static QString parseParkingSpaceAccessBack(ParkingSpace::ParkingSpaceAccess access);

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit ParkingSpace(ParkingSpaceAccess access, const QString &restrictions);
    virtual ~ParkingSpace()
    { /* does nothing */
    }

    // Object //
    //
	Object *getParentObject() const
	{
		return parentObject_;
	}
	void setParentObject(Object *parentObject);

	ParkingSpaceAccess getAccess() const
    {
        return access_;
    }
	void setAccess(ParkingSpaceAccess access);

	QString getRestrictions()
	{
		return restrictions_;
	}
	void setRestrictions(const QString &restrictions);

	void addMarking(ParkingSpaceMarking *marking);
	QMap<ParkingSpaceMarking::ParkingSpaceMarkingSide, ParkingSpaceMarking *> getMarkings()
	{
		return markingList_;
	}


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getParkingSpaceChanges() const
    {
        return parkingSpaceChanges_;
    }
    void addParkingSpaceChanges(int changes);


    // Prototype Pattern //
    //
    ParkingSpace *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    ParkingSpace(); /* not allowed */
    ParkingSpace(const ParkingSpace &); /* not allowed */
    ParkingSpace &operator=(const ParkingSpace &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // ParkingSpace //
    //
    // Mandatory
	ParkingSpaceAccess access_;
    QString restrictions_;

	Object *parentObject_;

    // Change flags //
    //
    int parkingSpaceChanges_;

	// Marking list //
	//
	QMap<ParkingSpaceMarking::ParkingSpaceMarkingSide, ParkingSpaceMarking *> markingList_;
};

#endif // PARKINGSPACEOBJECT_HPP
