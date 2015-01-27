/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    SetObject
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

#ifndef SET_OBJECT_H
#define SET_OBJECT_H

#include "OutputObject.h"
#include <string>
#include <vector>

class SetObject : public OutputObject
{
public:
    SetObject();

    virtual ~SetObject();

    virtual bool process(const int &fd);

    virtual SetObject *clone() const;

protected:
    SetObject(const SetObject &o);
    SetObject(const OutputObject &o);
};

#endif
