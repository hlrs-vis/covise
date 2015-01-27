/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHANGE_ATTRIBUTE_H
#define _CHANGE_ATTRIBUTE_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coModule.h>
using namespace covise;

////// our class
class ChangeAttribute : public coModule
{
private:
    /////////ports
    coInputPort *p_geo_in;
    coOutputPort *p_geo_out;

    coStringParam *p_rmvAttr, *p_chAttr, *p_chAttrVal;

    virtual int compute();

    void copyAttributes(coDistributedObject *tgt, coDistributedObject *src);
    coDistributedObject *createNewObj(const char *name, coDistributedObject *obj) const;

public:
    ChangeAttribute();
};
#endif
