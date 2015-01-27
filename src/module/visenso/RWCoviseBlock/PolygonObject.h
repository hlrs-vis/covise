/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    PolygonObject
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

#ifndef POLYGON_OBJECT_H
#define POLYGON_OBJECT_H

#include "OutputObject.h"
#include <string>
#include <vector>

class PolygonObject : public OutputObject
{
public:
    PolygonObject();

    virtual ~PolygonObject();

    virtual bool process(const int &fd);

    virtual PolygonObject *clone() const;

protected:
    PolygonObject(const PolygonObject &o);
    PolygonObject(const OutputObject &o);
};

#endif
