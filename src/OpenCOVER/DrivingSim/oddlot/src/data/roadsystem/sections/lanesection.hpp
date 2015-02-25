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

#ifndef LANESECTION_HPP
#define LANESECTION_HPP

#include "roadsection.hpp"

#include <QMap>

class LaneSection : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    // Observer Pattern //
    //
    enum LaneSectionChange
    {
        CLS_LanesChanged = 0x1,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSection(double s);
    explicit LaneSection(double s, const LaneSection *oldLaneSection); // create new LaneSection at pos s of oldLaneSection
    explicit LaneSection(double s, const LaneSection *LanSectionLow, const LaneSection *LanSectionHigh); // create new LaneSection as e merger between low and high
    virtual ~LaneSection();

    // Section Functions //
    //
    virtual double getSEnd() const;
    virtual double getLength() const;
    Lane *getNextLower(int id) const;
    Lane *getNextUpper(int id) const;
    Lane *getFirst() const;
    Lane *getLast() const;

    // Lane Functions //
    //
    const QMap<int, Lane *> &getLanes() const
    {
        return lanes_;
    }
    void setLanes(QMap<int, Lane *>);
    void addLane(Lane *lane);
    void removeLane(Lane *lane);
    Lane *getLane(int id) const;
    int getLaneId(double s, double t);

    double getLaneWidth(int lane, double s) const;
    double getLaneSpanWidth(int fromLane, int toLane, double s) const;
    int getLeftmostLaneId() const;
    int getRightmostLaneId() const;
    void checkAndFixLanes();

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLaneSectionChanges() const
    {
        return laneSectionChanges_;
    }

    // Prototype Pattern //
    //
    LaneSection *getClone() const;
    LaneSection *getClone(double sStart, double sEnd) const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForLanes(Visitor *visitor);

protected:
private:
    LaneSection(); /* not allowed */
    LaneSection(const LaneSection &); /* not allowed */
    LaneSection &operator=(const LaneSection &); /* not allowed */

    // Observer Pattern //
    //
    void addLaneSectionChanges(int changes);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneSectionChanges_;

protected:
    // Lane Entries //
    //
    QMap<int, Lane *> lanes_; // owned
};

#endif // LANESECTION_HPP
