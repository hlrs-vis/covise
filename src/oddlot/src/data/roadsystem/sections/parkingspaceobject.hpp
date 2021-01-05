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

class ParkingSpace 
{
	// nested class for marking defintition
	//

	class ParkingSpaceMarking
	{
	public:


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
		explicit ParkingSpaceMarking(Object *parentObject, ParkingSpaceMarkingSide side, LaneRoadMark::RoadMarkType type, double width, LaneRoadMark::RoadMarkColor color);

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


		// Prototype Pattern //
		//
		ParkingSpaceMarking *getClone();


	private:
		Object *parentObject_;
		ParkingSpace *parentParkingSpace_;

		ParkingSpaceMarkingSide side_;
		LaneRoadMark::RoadMarkType type_;
		double width_;
		LaneRoadMark::RoadMarkColor color_;
	};

    //################//
    // STATIC         //
    //################//

public:

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
	explicit ParkingSpace(Object *parentObject,  ParkingSpaceAccess access, const QString &restrictions);
    virtual ~ParkingSpace()
    { /* does nothing */
    }

    // Object //
    //
	void setParentObject(Object *parentObject);
	Object *getParentObject() const
	{
		return parentObject_;
	}

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
	bool addMarking(QString side, QString type, double width, QString color);
	int getMarkingsSize()
	{
		return markingList_.size();
	}
	bool getMarking(int i, QString &side, QString &type, double &width, QString &color);



    // Prototype Pattern //
    //
    ParkingSpace *getClone();


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
