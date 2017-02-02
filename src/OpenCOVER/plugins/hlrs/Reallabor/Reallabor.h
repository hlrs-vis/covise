/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REALLABOR_H
#define REALLABOR_H

#include <cover/coVRPlugin.h>

class Reallabor : public opencover::coVRPlugin
{
public:
    Reallabor();
    ~Reallabor();

    // this will be called in PreFrame
    void preFrame();

private:
};
#endif
