/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                           (C)2009 UKOELN **
 **                                                                          **
 ** Description: Background Plugin                                           **
 **                                                                          **
 **                                                                          **
 ** Author: D. Wickeroth	                                               **
 **                                                                          **
 ** History:                                                                 **
 ** Dez 2009  v1                                                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Background.h"
#include <cover/coVRPluginSupport.h>

#include "coVRConfig.h"
#include <config/coConfig.h>
#include <config/CoviseConfig.h>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>

// Virvo:
#include <virvo/vvtoolshed.h>

// OSG:
#include <osg/Vec4>

using namespace opencover;

Background::Background()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "\nBackground Plugin\n");

    hue = 0.0f;
    saturation = 0.0f;
    brightness = 0.0f;

    default_hue = 0.0f;
    default_saturation = 0.0f;
    default_brightness = 0.0f;

    useDefault = false;
}

Background::~Background()
{
    config->setValue("h", QString("%1").arg(hue), "COVER.Background");
    config->setValue("s", QString("%1").arg(saturation), "COVER.Background");
    config->setValue("b", QString("%1").arg(brightness), "COVER.Background");

    config->save();
    coConfig::getInstance()->removeConfig(config->getGroupName());

    delete config;

    removeMenuEntry();
}

bool Background::init()
{

    //check if a bg-color has been set in the standard config files
    //use as default if so
    float r = coCoviseConfig::getFloat("r", "COVER.Background", -1.0f);
    float g = coCoviseConfig::getFloat("g", "COVER.Background", -1.0f);
    float b = coCoviseConfig::getFloat("b", "COVER.Background", -1.0f);

    if ((r >= 0.0f) & (g >= 0.0f) & (b >= 0.0f))
    {
        vvToolshed::RGBtoHSB(r, g, b, &default_hue, &default_saturation, &default_brightness);
    }

    //get background color from own config
    config = new coConfigGroup("BackgroundPlugin");
    config->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "BackgroundPlugin.xml", "local", true);
    coConfig::getInstance()->addConfig(config);

    hue = getConfigValue("h");
    saturation = getConfigValue("s");
    brightness = getConfigValue("b");

    fprintf(stderr, "hue : %f   saturation %f   brightness : %f ", hue, saturation, brightness);
    setClearColor();

    createMenuEntry();

    return true;
}

void Background::createMenuEntry()
{

    // create new background menu
    backgroundMenuItem = new coSubMenuItem("Background");

    // create row menu to add items
    backgroundMenu = new coRowMenu("Background Options");
    backgroundMenuItem->setMenu(backgroundMenu);

    // use colored backgrounds
    backgroundMenuCheckbox = new coCheckboxMenuItem("Use colored background", false);
    backgroundMenu->add(backgroundMenuCheckbox);
    backgroundMenuCheckbox->setMenuListener(this);
    backgroundMenuCheckbox->setState(true);

    //choose color
    backgroundSliderHue = new coSliderMenuItem("Hue", 0.0f, 1.0f, 0.0f);
    backgroundMenu->add(backgroundSliderHue);
    backgroundSliderHue->setMenuListener(this);
    backgroundSliderHue->setValue(hue);

    backgroundSliderSaturation = new coSliderMenuItem("Saturation", 0.0f, 1.0f, 0.0f);
    backgroundMenu->add(backgroundSliderSaturation);
    backgroundSliderSaturation->setMenuListener(this);
    backgroundSliderSaturation->setValue(saturation);

    backgroundSliderBrightness = new coSliderMenuItem("Brightness", 0.0f, 1.0f, 0.0f);
    backgroundMenu->add(backgroundSliderBrightness);
    backgroundSliderBrightness->setMenuListener(this);
    backgroundSliderBrightness->setValue(brightness);

    cover->getMenu()->add(backgroundMenuItem);
}

void Background::removeMenuEntry()
{
    delete backgroundMenuItem;
}

void Background::menuEvent(coMenuItem *item)
{

    if (item == backgroundMenuCheckbox)
    {
        bool state = backgroundMenuCheckbox->getState();
        useDefault = !state;
        setClearColor();
    }

    else if (item == backgroundSliderHue)
    {
        hue = backgroundSliderHue->getValue();
        setClearColor();
    }

    else if (item == backgroundSliderSaturation)
    {
        saturation = backgroundSliderSaturation->getValue();
        setClearColor();
    }

    else if (item == backgroundSliderBrightness)
    {
        brightness = backgroundSliderBrightness->getValue();
        setClearColor();
    }
}

void Background::setClearColor()
{
    float r, g, b;
    osg::Vec4 color;

    if (!useDefault)
    {
        vvToolshed::HSBtoRGB(hue, saturation, brightness, &r, &g, &b);
    }
    else
    {
        vvToolshed::HSBtoRGB(default_hue, default_saturation, default_brightness, &r, &g, &b);
    }

    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = 1.0;

    for (int i = 0; i < coVRConfig::instance()->numScreens(); ++i)
    {
        coVRConfig::instance()->screens[i].camera->setClearColor(color);
    }
}

float Background::getConfigValue(std::string name)
{

    float result = 0.0f;

    if (config != 0)
    {
        coConfigEntryString ces;
        ces = config->getValue(name.c_str(), "COVER.Background", QString("%1").arg(0.0));

        bool ok;
        float f = ces.toFloat(&ok);
        if (ok)
            result = f;
    }

    return result;
}

COVERPLUGIN(Background)
