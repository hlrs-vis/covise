/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   1/18/2010
**
**************************************************************************/

#include "trackelement.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//################//
// CONSTRUCTOR    //
//################//

TrackElement::TrackElement(double x, double y, double angleDegrees, double s, double length)
    : TrackComponent(x, y, angleDegrees)
    , s_(s)
    , length_(length)
{
}

TrackElement::~TrackElement()
{
}

//################//
// TRACK ELEMENT  //
//################//

void
TrackElement::setSStart(double s)
{
    s_ = s;
    addTrackComponentChanges(TrackComponent::CTC_SChange);
}

void
TrackElement::setLength(double length)
{
    length_ = length;
    addTrackComponentChanges(TrackComponent::CTC_LengthChange);
}
