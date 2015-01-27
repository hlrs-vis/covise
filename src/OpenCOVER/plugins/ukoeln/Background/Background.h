/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BACKGROUND_PLUGIN_H
#define _BACKGROUND_PLUGIN_H
/****************************************************************************\ 
 **                                                           (C)2009 UKOELN **
 **                                                                          **
 ** Description: Background Plugin                                           **
 **                                                                          **
 **                                                                          **
 ** Author: D. Wickeroth                                                     **
 **                                                                          **
 ** History:                                                                 **
 ** Dez 2009  v1                                                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

#include <OpenVRUI/coMenuItem.h>

namespace covise
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coSliderMenuItem;
class coConfigGroup;
}

using namespace covise;
using namespace opencover;

class Background : public coVRPlugin, public coMenuListener
{
public:
    static Background *plugin;

    Background();

    virtual ~Background();

    bool init();

private:
    // The VR Menu Interface
    void createMenuEntry(); ///< create a VR menu item "Annotations"
    void removeMenuEntry(); ///< remove the VR menu item
    void menuEvent(coMenuItem *); ///< handles VR menu events

    void setClearColor();

    float getConfigValue(std::string name);

    coConfigGroup *config;

    coRowMenu *backgroundMenu;
    coSubMenuItem *backgroundMenuItem;
    coCheckboxMenuItem *backgroundMenuCheckbox;
    coSliderMenuItem *backgroundSliderHue;
    coSliderMenuItem *backgroundSliderSaturation;
    coSliderMenuItem *backgroundSliderBrightness;

    float hue;
    float saturation;
    float brightness;

    float default_hue;
    float default_saturation;
    float default_brightness;

    bool useDefault;
};
#endif //_BACKGROUND_PLUGIN_H
