/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <sstream>
using namespace std;

class HelpFuncs
{
public:
    HelpFuncs(void);
    ~HelpFuncs(void);
    static void IntToString(int i, string &res);
    static void FloatToString(float i, string &res);
};
