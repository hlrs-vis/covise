/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TESTINGMODULE_H
#define TESTINGMODULE_H

#include <api/coModule.h>

class testingModule : public covise::coModule
{
public:
    testingModule(int argc, char *argv[]);
    virtual int compute(char const *str);
    virtual ~testingModule();
};
#endif /* TESTINGMODULE_H */
