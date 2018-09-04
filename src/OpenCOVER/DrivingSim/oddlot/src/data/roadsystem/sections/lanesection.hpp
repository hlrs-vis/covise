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
		CLS_LanesWidthsChanged = 0x2
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSection(double s, bool singleSide);
    explicit LaneSection(double s, bool singleSide, const LaneSection *oldLaneSection); // create new LaneSection at pos s of oldLaneSection
    explicit LaneSection(double s, bool singleSide, const LaneSection *LanSectionLow, const LaneSection *LanSectionHigh); // create new LaneSection as e merger between low and high
    virtual ~LaneSection();

	bool getSide()
	{
		return singleSide_;
	}

	void setSide(bool singleSide)
	{
		singleSide_ = singleSide;
	}

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

    // Returns distance from road center to mid of lane
    //
    double getTValue(Lane * lane, double s, double laneWidth);

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

	// Observer Pattern //
	//
	void addLaneSectionChanges(int changes);

protected:
private:
    LaneSection(); /* not allowed */
    LaneSection(const LaneSection &); /* not allowed */
    LaneSection &operator=(const LaneSection &); /* not allowed */


    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int laneSectionChanges_;

	// valid for only one side //
	// 
	bool singleSide_;

protected:
    // Lane Entries //
    //
    QMap<int, Lane *> lanes_; // owned
};

#endif // LANESECTION_HPP
