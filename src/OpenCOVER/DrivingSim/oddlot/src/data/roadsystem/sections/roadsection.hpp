/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   18.03.2010
**
**************************************************************************/

#ifndef ROADSECTION_HPP
#define ROADSECTION_HPP

#include "src/data/dataelement.hpp"

class RoadSection : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum RoadSectionChange
    {
        CRS_SChange = 0x1,
        CRS_LengthChange = 0x2,
        CRS_ParentRoadChange = 0x4
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSection(double s);
    virtual ~RoadSection();

    // Section //
    //
    virtual double getSStart() const
    {
        return s_;
    }
    virtual double getSEnd() const = 0;
    virtual double getLength() const = 0;

    void setSStart(double s);

    // Road //
    //
    RSystemElementRoad *getParentRoad() const
    {
        return parentRoad_;
    }
    void setParentRoad(RSystemElementRoad *parentRoad);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getRoadSectionChanges() const
    {
        return roadSectionChanges_;
    }
    void addRoadSectionChanges(int changes);

private:
    RoadSection(); /* not allowed */
    RoadSection(const RoadSection &); /* not allowed */
    RoadSection &operator=(const RoadSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
    RSystemElementRoad *parentRoad_; // linked

    // Section //
    //
    double s_;

    // Change //
    //
    int roadSectionChanges_;
};

#endif // ROADSECTION_HPP
