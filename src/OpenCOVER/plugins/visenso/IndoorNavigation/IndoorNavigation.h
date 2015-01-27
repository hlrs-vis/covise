/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2013 Visenso  **
 **                                                                        **
 ** Description: Indoor Navigation (HSG-IMIT)                              **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
\****************************************************************************/

#ifndef _INDOOR_NAVIGATION_H
#define _INDOOR_NAVIGATION_H

#include "Path.h"
#include "Avatar.h"

#include <cover/coVRPlugin.h>
using namespace covise;
using namespace opencover;

class IndoorNavigation : public coVRPlugin
{
public:
    // constructor destructor
    IndoorNavigation();
    virtual ~IndoorNavigation();

    // variables of class
    static IndoorNavigation *plugin;

    virtual bool init();
    virtual void preFrame();
    virtual void key(int type, int keySym, int mod);

private:
    osg::ref_ptr<Path> path;
    osg::ref_ptr<Avatar> avatar;
    osg::ref_ptr<osg::Group> pluginBaseNode;

    float animationSeconds;

    // settings
    float speed;
    float clipOffset;
    std::string filename;
};

#endif
