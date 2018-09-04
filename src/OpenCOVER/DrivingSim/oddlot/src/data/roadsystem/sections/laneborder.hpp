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

#ifndef LANEBORDER_HPP
#define LANEBORDER_HPP

#include "src/data/roadsystem/sections/lanewidth.hpp"

/**
* NOTE: s = s_ + sOffset + ds
* sSection (= sOffset + ds) is relative to s_ of the laneSection
*/
class LaneBorder : public LaneWidth
{

public:

    //################//
    // FUNCTIONS      //
    //################//

public:
	// Observer Pattern //
	//
	enum LaneBorderChange
	{
		CLB_BorderChanged = 0x1
	};

    explicit LaneBorder(double sOffset, double a, double b, double c, double d);
    virtual ~LaneBorder();

	LaneBorder *getClone();

	double getT(double s);

    // Observer Pattern //
    //
    virtual void notificationDone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    LaneBorder(); /* not allowed */
    LaneBorder(const LaneBorder &); /* not allowed */
    LaneBorder &operator=(const LaneBorder &); /* not allowed */

public:

    //################//
    // PROPERTIES     //
    //################//

private:
	int laneBorderChanges_;

};

#endif // LANEBORDER_HPP
