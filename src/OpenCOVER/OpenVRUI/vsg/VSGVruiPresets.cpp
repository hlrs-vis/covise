/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiPresets.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiLog.h>
#include <config/CoviseConfig.h>

#include <vsg/io/Options.h>
#include <vsg/io/read.h>
#include <vsg/state/material.h>
#include <vsgXchange/all.h>


namespace vrui
{

using covise::coCoviseConfig;

VSGVruiPresets *VSGVruiPresets::s_instance = nullptr;

VSGVruiPresets::VSGVruiPresets()
{
    materials[coUIElement::RED].ambient = vsg::vec4(0.2f, 0.0f, 0.0f, 1.0f);
    materials[coUIElement::RED].diffuse = vsg::vec4(1.0f, 0.0f, 0.0f, 1.0f);

    materials[coUIElement::GREEN].ambient = vsg::vec4(0.0f, 0.2f, 0.0f, 1.0f);
    materials[coUIElement::GREEN].diffuse = vsg::vec4(0.0f, 1.0f, 0.0f, 1.0f);

    materials[coUIElement::BLUE].ambient = vsg::vec4(0.0f, 0.0f, 0.2f, 1.0f);
    materials[coUIElement::BLUE].diffuse = vsg::vec4(0.0f, 0.0f, 1.0f, 1.0f);

    materials[coUIElement::YELLOW].ambient = vsg::vec4(0.2f, 0.2f, 0.0f, 1.0f);
    materials[coUIElement::YELLOW].diffuse = vsg::vec4(1.0f, 1.0f, 0.0f, 1.0f);

    materials[coUIElement::GREY].ambient = vsg::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    materials[coUIElement::GREY].diffuse = vsg::vec4(0.1f, 0.1f, 0.1f, 1.0f);

    materials[coUIElement::WHITE].ambient = vsg::vec4(0.3f, 0.3f, 0.3f, 1.0f);
    materials[coUIElement::WHITE].diffuse = vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f);

    materials[coUIElement::BLACK].ambient = vsg::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    materials[coUIElement::BLACK].diffuse = vsg::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    materials[coUIElement::DARK_YELLOW].ambient = vsg::vec4(0.3f, 0.5f, 0.0f, 1.0f);
    materials[coUIElement::DARK_YELLOW].diffuse = vsg::vec4(0.3f, 0.5f, 0.0f, 1.0f);

    materials[coUIElement::WHITE_NL].ambient = vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    materials[coUIElement::WHITE_NL].diffuse = vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    materials[coUIElement::WHITE_NL].emissive = vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f);

    setColorFromConfig("COVER.VRUI.ItemBackgroundNormal", coUIElement::ITEM_BACKGROUND_NORMAL, vsg::vec4(0.1f, 0.1f, 0.1f, 1.0f));
    setColorFromConfig("COVER.VRUI.ItemBackgroundHighlighted", coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vsg::vec4(0.3f, 0.5f, 0.0f, 1.0f));
    setColorFromConfig("COVER.VRUI.ItemBackgroundDisabled", coUIElement::ITEM_BACKGROUND_DISABLED, vsg::vec4(0.6f, 0.6f, 0.6f, 1.0f));
    setColorFromConfig("COVER.VRUI.HandleBackgroundNormal", coUIElement::HANDLE_BACKGROUND_NORMAL, vsg::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    setColorFromConfig("COVER.VRUI.HandleBackgroundHighlighted", coUIElement::HANDLE_BACKGROUND_HIGHLIGHTED, vsg::vec4(0.6f, 0.6f, 0.0f, 1.0f));
    setColorFromConfig("COVER.VRUI.HandleBackgroundDisabled", coUIElement::HANDLE_BACKGROUND_DISABLED, vsg::vec4(0.6f, 0.6f, 0.6f, 1.0f));

    setColorFromConfig("COVER.Background", coUIElement::BACKGROUND, vsg::vec4(9.f, 0.f, 0.f, 1.f));
    options = vsg::Options::create();
    auto fontName = coCoviseConfig::getEntry("value", "COVER.VRUI.Font", "times.vsgb");

    fontFile = vruiRendererInterface::the()->getFont(fontName);
    std::string fontFileName  = vruiRendererInterface::the()->getFont(fontName); 
    font = vsg::read_cast<vsg::Font>(fontFileName, options);
    if (!font)
    {
        std::cout << "Failed to read font : " << fontFileName << std::endl;
    }
    sharedObjects = vsg::SharedObjects::create();
}

VSGVruiPresets::~VSGVruiPresets()
{
}



void VSGVruiPresets::setColorFromConfig(const char *configEntry, int materialIndex, vsg::vec4 def)
{
    vsg::vec4 color;
    color = vsg::vec4(coCoviseConfig::getFloat("r", configEntry, def[0]),
                      coCoviseConfig::getFloat("g", configEntry, def[1]),
                      coCoviseConfig::getFloat("b", configEntry, def[2]), 1.0f);

    vsg::PhongMaterial mat;

    materials[materialIndex].ambient = color;
    materials[materialIndex].diffuse = color;
}
}
