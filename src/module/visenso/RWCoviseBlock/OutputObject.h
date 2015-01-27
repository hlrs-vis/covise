/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    OutputObject
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

#ifndef OUTPUTOBJECT_H
#define OUTPUTOBJECT_H

#include "covise/covise.h"
#include <string>
#include <stdio.h>
#include <do/coDistributedObject.h>

using namespace covise;

// REMARK: It would be really cool to make this object reference-counted
//         as it will be exclusively created by a factory and therefore
//         it should take care of its death by itself.
//
class OutputObject
{
public:
    OutputObject();

    std::string type() const;
    virtual ~OutputObject(){};

    //virtual bool process();
    virtual bool process(const int &fd);
    virtual OutputObject *clone() const;

    void setDO(const coDistributedObject *d);

protected:
    // create a copy of the minimally initialized object o
    // which has the property prop set.
    //OutputObject *burnIn(const int &prop, const OutputObject *o);

    OutputObject(const OutputObject &o);
    OutputObject(const std::string &type);
    std::string type_;

    const coDistributedObject *distrObj_;
};

#endif
