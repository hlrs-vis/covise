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

#ifndef LANEWIDTH_HPP
#define LANEWIDTH_HPP

#include "src/data/dataelement.hpp"
#include "src/util/math/polynomial.hpp"

/**
* NOTE: s = s_ + sOffset + ds
* sSection (= sOffset + ds) is relative to s_ of the laneSection
*/
class LaneWidth : public DataElement, public Polynomial
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneWidthChange
    {
        CLW_ParentLaneChanged = 0x1,
        CLW_OffsetChanged = 0x2,
        CLW_WidthChanged = 0x4,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneWidth(double sOffset, double a, double b, double c, double d);
    virtual ~LaneWidth();

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
    double getSSectionStartAbs() const;
    double getSSectionEnd() const;
    double getLength() const;

    double getSOffset() const
    {
        return sOffset_;
    }
    void setSOffset(double sOffset);

    double getWidth(double sSection) const;
    double getSlope(double sSection) const;
    double getCurvature(double sSection) const;

    void setParameters(double a, double b, double c, double d);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLaneWidthChanges() const
    {
        return laneWidthChanges_;
    }

    // Prototype Pattern //
    //
    LaneWidth *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneWidth(); /* not allowed */
    LaneWidth(const LaneWidth &); /* not allowed */
    LaneWidth &operator=(const LaneWidth &); /* not allowed */

public:
    // Observer Pattern //
    //
    void addLaneWidthChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneWidthChanges_;

    // Lane Properties //
    //
    Lane *parentLane_; // linked

    double sOffset_;
};

#endif // LANEWIDTH_HPP
