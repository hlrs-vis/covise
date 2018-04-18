/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.02.2010
**
**************************************************************************/

#ifndef LANEOFFSET_HPP
#define LANEOFFSET_HPP

#include "src/data/dataelement.hpp"
#include "src/util/math/polynomial.hpp"

class RSystemElementRoad;

/**
* NOTE: s = s_ + sOffset + ds
* sSection (= sOffset + ds) is relative to s_ of the laneSection
*/
class LaneOffset : public DataElement, public Polynomial
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneOffsetChange
    {
        CLO_ParentRoadChanged = 0x1,
		CLO_OffsetChanged = 0x2,
		CLO_SOffsetChanged = 0x3,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneOffset(double sOffset, double a, double b, double c, double d);
    virtual ~LaneOffset();

    // Lane Functions //
    //
	RSystemElementRoad *getParentRoad() const
    {
        return parentRoad_;
    }
    void setParentRoad(RSystemElementRoad *parentRoad);

    double getSSectionStart() const
    {
        return sOffset_;
    }

	double getSOffset() const
	{
		return sOffset_;
	}
	double getSStart() const
	{
		return sOffset_;
	}
    void setSOffset(double sOffset);

    double getOffset(double sSection) const;
    double getSlope(double sSection) const;
    double getCurvature(double sSection) const;

    void setParameters(double a, double b, double c, double d);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLaneOffsetChanges() const
    {
        return laneOffsetChanges_;
    }

    // Prototype Pattern //
    //
    LaneOffset *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneOffset(); /* not allowed */
    LaneOffset(const LaneOffset &); /* not allowed */
    LaneOffset &operator=(const LaneOffset &); /* not allowed */

public:
    // Observer Pattern //
    //
    void addLaneOffsetChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneOffsetChanges_;

    // Lane Properties //
    //
    RSystemElementRoad *parentRoad_; // linked

    double sOffset_;
};

#endif // LANEOFFSET_HPP
