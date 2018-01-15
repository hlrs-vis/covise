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

#ifndef LATERALSECTIONCOMMANDS_HPP
#define LATERALSECTIONCOMMANDS_HPP


#include "datacommand.hpp"

class LateralSection;

//###############//
// MoveLateralSectionCommand //
//###############//

class MoveLateralSectionCommand : public DataCommand
{
public:
    explicit MoveLateralSectionCommand(LateralSection *lateralSection, double t, DataCommand *parent = NULL);
    virtual ~MoveLateralSectionCommand();

    virtual int id() const
    {
        return 0x411;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    MoveLateralSectionCommand(); /* not allowed */
    MoveLateralSectionCommand(const MoveLateralSectionCommand &); /* not allowed */
    MoveLateralSectionCommand &operator=(const MoveLateralSectionCommand &); /* not allowed */

private:
	LateralSection *lateralSection_;

    double newT_;
    double oldT_;
};

#endif // LATERALSECTIONCOMMANDS_HPP
