/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ADDATTRIBUTE_H
#define ADDATTRIBUTE_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 ZAIK ++
// ++ Description: Add attribute                                          ++
// ++                                                                     ++
// ++ Author: Martin Aumueller (aumueller@uni-koeln.de)                   ++
// ++                                                                     ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;

class AddAttribute : public coModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    int recurse(const coDistributedObject *obj);

    ////////// ports
    coInputPort *p_inPort;
    coOutputPort *p_outPort;
    coStringParam *p_attrName;
    coStringParam *p_attrVal;

public:
    AddAttribute(int argc, char *argv[]);
};
#endif
