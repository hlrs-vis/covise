/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    ScalDataObject
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

#ifndef SCALDATA_OBJECT_H
#define SCALDATA_OBJECT_H

#include "OutputObject.h"
#include <string>
#include <vector>

class ScalDataObject : public OutputObject
{

public:
    ScalDataObject();

    virtual ~ScalDataObject();

    virtual bool process(const int &fd);

    virtual ScalDataObject *clone() const;

protected:
    ScalDataObject(const ScalDataObject &o);
};

#endif
