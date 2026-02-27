/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LOGO_PLUGIN_H
#define _LOGO_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <util/common.h>

#include <osg/Geode>
#include <osg/Version>

using namespace opencover;

#include "Logo.h"

// displays a bitmap logo
class LogoPlugin : public coVRPlugin
{
public:
    LogoPlugin();
    ~LogoPlugin() override;
    bool init() override;
    bool destroy() override;

    void preFrame() override;
    void message(int toWhom, int type, int length, const void *data) override;

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
