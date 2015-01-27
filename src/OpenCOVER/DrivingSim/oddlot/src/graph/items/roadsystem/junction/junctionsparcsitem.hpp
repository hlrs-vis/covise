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

#include "junctioncompositeitem.hpp"

class TrackSpiralArcSpiral;

class TrackElementSpiral;
class TrackElementArc;
class TrackElementSpiral;

class JunctionSparcsHandle;

class JunctionSpArcSItem : public JunctionCompositeItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionSpArcSItem(JunctionRoadItem *parentJunctionRoadItem, TrackSpiralArcSpiral *trackSpArcS);
    explicit JunctionSpArcSItem(JunctionComponentItem *parentJunctionComponentItem, TrackSpiralArcSpiral *trackSpArcS);
    virtual ~JunctionSpArcSItem();

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
    JunctionSpArcSItem(); /* not allowed */
    JunctionSpArcSItem(const JunctionSpArcSItem &); /* not allowed */
    JunctionSpArcSItem &operator=(const JunctionSpArcSItem &); /* not allowed */

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
    JunctionSparcsHandle *factorHandle_;
};

#endif // SPARCSITEM_HPP
