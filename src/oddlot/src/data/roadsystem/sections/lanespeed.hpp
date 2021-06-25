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

#ifndef LANESPEED_HPP
#define LANESPEED_HPP

#include "src/data/dataelement.hpp"

class LaneSpeed : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneSpeedChange
    {
        CLS_ParentLaneChanged = 0x1,
        CLS_OffsetChanged = 0x2,
        CLS_MaxSpeedChanged = 0x4,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSpeed(double sOffset, double max, QString unit);
    virtual ~LaneSpeed();

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

    // Parameters //
    //
    double getSOffset() const
    {
        return sOffset_;
    }
    void setSOffset(double sOffset);

    double getMaxSpeed() const
    {
        return max_;
    }
    void setMaxSpeed(double max);

	QString getMaxSpeedUnit()
	{
		return maxUnit_;
	}

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLaneSpeedChanges() const
    {
        return laneSpeedChanges_;
    }

    // Prototype Pattern //
    //
    LaneSpeed *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneSpeed(); /* not allowed */
    LaneSpeed(const LaneSpeed &); /* not allowed */
    LaneSpeed &operator=(const LaneSpeed &); /* not allowed */

    // Observer Pattern //
    //
    void addLaneSpeedChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneSpeedChanges_;

    // Lane Properties //
    //
    Lane *parentLane_; // linked

    double sOffset_;
    double max_;
	QString maxUnit_;
};

#endif // LANESPEED_HPP
