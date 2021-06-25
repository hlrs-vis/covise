/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.04.2010
**
**************************************************************************/

#ifndef SPARCSITEM_HPP
#define SPARCSITEM_HPP

#include "trackcompositeitem.hpp"

class TrackSpiralArcSpiral;

class TrackElementSpiral;
class TrackElementArc;
class TrackElementSpiral;

class TrackSparcsHandle;

class TrackSpArcSItem : public TrackCompositeItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackSpArcSItem(TrackRoadItem *parentTrackRoadItem, TrackSpiralArcSpiral *trackSpArcS);
    explicit TrackSpArcSItem(TrackComponentItem *parentTrackComponentItem, TrackSpiralArcSpiral *trackSpArcS);
    virtual ~TrackSpArcSItem();

    // Track //
    //
    TrackSpiralArcSpiral *getSpArcS() const
    {
        return trackSpArcS_;
    }

    // Graphics //
    //
    virtual void updateColor()
    {
        ;
    }
    virtual void updateFactorHandle();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    TrackSpArcSItem(); /* not allowed */
    TrackSpArcSItem(const TrackSpArcSItem &); /* not allowed */
    TrackSpArcSItem &operator=(const TrackSpArcSItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Track //
    //
    TrackSpiralArcSpiral *trackSpArcS_;

    TrackElementSpiral *inSpiral_;
    TrackElementArc *arc_;
    TrackElementSpiral *outSpiral_;

    // Slider //
    //
    TrackSparcsHandle *factorHandle_;
};

#endif // SPARCSITEM_HPP
