/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.02.2010
**
**************************************************************************/

#ifndef LANEACCESS_HPP
#define LANEACCESS_HPP

#include "src/data/dataelement.hpp"

#include <QString>

class LaneAccess : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneAccessChange
    {
        CLA_ParentLaneChanged = 0x1,
        CLA_OffsetChanged = 0x2,
        CLA_RestrictionChanged = 0x4
    };

	enum LaneRestriction
	{
		LAR_UNKNOWN,
		LAR_SIMULATOR,
		LAR_AUTONOMOUS,
		LAR_PEDESTRIAN,
		LAR_NONE
	};

	static LaneRestriction parseLaneRestriction(const QString &restriction);
	static QString parseLaneRestrictionBack(LaneRestriction restriction);


    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneAccess(double sOffset, LaneRestriction restriction);
    virtual ~LaneAccess();

    // Lane Functions //
    //
    Lane *getParentLane() const
    {
        return parentLane_;
    }
    void setParentLane(Lane *parentLane);

    double getSSectionStart() const
    {
        return sOffset_;
    }
    double getSSectionEnd() const;
    double getLength() const;

    // RoadMark Parameters //
    //
    double getSOffset() const
    {
        return sOffset_;
    }
    void setSOffset(double sOffset);

	LaneRestriction getRestriction() const
    {
        return restriction_;
    }
    void setRestriction(LaneRestriction restriction);


    int getAccessChanges() const
    {
        return accessChanges_;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();

    // Prototype Pattern //
    //
    LaneAccess *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneAccess(); /* not allowed */
    LaneAccess(const LaneAccess &); /* not allowed */
    LaneAccess &operator=(const LaneAccess &); /* not allowed */

    // Observer Pattern //
    //
    void addAccessChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int accessChanges_;

    // Lane Properties //
    //
    Lane *parentLane_; // linked

    // RoadMark Properties //
    //
    double sOffset_;
	LaneRestriction restriction_;
};

#endif // LANEACCESS_HPP
