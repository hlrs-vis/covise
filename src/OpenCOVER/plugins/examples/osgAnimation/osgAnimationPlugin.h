/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _osgAnimation_PLUGIN_H
#define _osgAnimation_PLUGIN_H

#include <cover/coVRPlugin.h>
using namespace opencover;

// skin Animation exaple
class osgAnimationPlugin : public coVRPlugin
{
public:
    osgAnimationPlugin();
    ~osgAnimationPlugin();

    // this will be called in PreFrame
    void preFrame();

private:
};
#endif
