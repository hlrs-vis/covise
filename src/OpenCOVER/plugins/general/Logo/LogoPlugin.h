/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LOGO_PLUGIN_H
#define _LOGO_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2009 HLRS  **
**                                                                          **
** Description: Logo Plugin (displays a bitmap logo)                        **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                  **
**                                                                          **
** History:  								                                         **
** Feb-09  v1	    				       		                                   **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <util/common.h>

#include <osg/Geode>
#include <osg/Version>

using namespace opencover;

#include "Logo.h"

class LogoPlugin : public coVRPlugin
{
public:
    LogoPlugin();
    ~LogoPlugin();
    virtual bool init();
    virtual bool destroy();

    void preFrame();
    virtual void message(int type, int length, const void *data);

private:
    Logo *defaultLogo;
    Logo *recordingLogo;
    osg::ref_ptr<osg::Camera> camera;
    double hudTime;
    double logoTime;
    bool doHide;
    bool hidden;
};
#endif
