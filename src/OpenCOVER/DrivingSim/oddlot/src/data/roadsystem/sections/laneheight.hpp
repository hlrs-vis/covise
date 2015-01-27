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

#ifndef LANEHEIGHT_HPP
#define LANEHEIGHT_HPP

#include "src/data/dataelement.hpp"

class LaneHeight : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneHeightChange
    {
        CLW_ParentLaneChanged = 0x1,
        CLW_OffsetChanged = 0x2,
        CLW_HeightChanged = 0x4,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneHeight(double sOffset, double inner, double outer);
    virtual ~LaneHeight();

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

    double getInnerHeight() const;
    double getOuterHeight() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLaneHeightChanges() const
    {
        return laneHeightChanges_;
    }

    // Prototype Pattern //
    //
    LaneHeight *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneHeight(); /* not allowed */
    LaneHeight(const LaneHeight &); /* not allowed */
    LaneHeight &operator=(const LaneHeight &); /* not allowed */

    // Observer Pattern //
    //
    void addLaneHeightChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneHeightChanges_;

    // Lane Properties //
    //
    Lane *parentLane_; // linked

    double sOffset_;
    double inner_;
    double outer_;
};

#endif // LANEHEIGHT_HPP
