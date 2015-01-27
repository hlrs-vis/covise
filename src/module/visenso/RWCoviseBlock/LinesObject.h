/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    LinesObject
//
// Description:
//
// Initial version: 04.2007
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2009 by VISENSO GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//     Ported: 05.2009
//

#ifndef LINES_OBJECT_H
#define LINES_OBJECT_H

#include "OutputObject.h"
#include <string>
#include <vector>

class LinesObject : public OutputObject
{
public:
    LinesObject();

    virtual ~LinesObject();

    virtual bool process(const int &fd);
    virtual LinesObject *clone() const;

protected:
    LinesObject(const LinesObject &o);
    LinesObject(const OutputObject &o);

private:
    // this should not be created
    std::vector<int> linesList_;
};

#endif
