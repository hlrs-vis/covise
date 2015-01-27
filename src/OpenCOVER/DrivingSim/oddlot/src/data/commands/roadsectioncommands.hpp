/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#ifndef ROADSECTIONCOMMANDS_HPP
#define ROADSECTIONCOMMANDS_HPP

// 400

#include "datacommand.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
class RoadSection;

//###############//
// MoveRoadSectionCommand //
//###############//

class MoveRoadSectionCommand : public DataCommand
{
public:
    explicit MoveRoadSectionCommand(RoadSection *roadSection, double s, RSystemElementRoad::DRoadSectionType sectionType, DataCommand *parent = NULL);
    virtual ~MoveRoadSectionCommand();

    virtual int id() const
    {
        return 0x401;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    MoveRoadSectionCommand(); /* not allowed */
    MoveRoadSectionCommand(const MoveRoadSectionCommand &); /* not allowed */
    MoveRoadSectionCommand &operator=(const MoveRoadSectionCommand &); /* not allowed */

private:
    RoadSection *roadSection_;

    double newS_;
    double oldS_;

    RSystemElementRoad::DRoadSectionType sectionType_;
};

#endif // ROADSECTIONCOMMANDS_HPP
