/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _HELLO_PLUGIN_H
#define _HELLO_PLUGIN_H

#include <cover/coVRPlugin.h>

class Hello : public opencover::coVRPlugin
{
public:
    Hello();
    ~Hello();
};
#endif
