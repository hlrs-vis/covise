/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_PRESETS
#define OSG_VRUI_PRESETS

#include <OpenVRUI/coUIElement.h>

#include <vsg/utils/GraphicsPipelineConfigurator.h>
#include <vsg/maths/vec4.h>
#include <vsg/state/material.h>
#include <vsg/text/Font.h>

#include <vector>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiPresets
{

public:
    enum Configurations
    {
        ColorOnly=0,
        Textured,
        numConfigurations
    };
    static VSGVruiPresets* instance() { if (!s_instance) s_instance = new VSGVruiPresets; return s_instance; };
    vsg::ref_ptr <vsg::ShaderSet> phongShaderSet;
    vsg::ref_ptr<vsg::ShaderSet> getOrCreatePhongShaderSet()
    {
        if (!phongShaderSet)
        {
            phongShaderSet = vsg::createPhongShaderSet(options);
        }
        return phongShaderSet;
    }
    std::vector<vsg::ref_ptr<vsg::DescriptorConfigurator>> configurations;
    vsg::PhongMaterial materials[coUIElement::NUM_MATERIALS];
    vsg::ref_ptr<const vsg::Options> options;
    std::string fontFile;

    vsg::ref_ptr<vsg::Font> font;

    vsg::ref_ptr<vsg::SharedObjects> sharedObjects;

private:
    VSGVruiPresets();
    virtual ~VSGVruiPresets();

    static VSGVruiPresets* s_instance;
    void setColorFromConfig(const char *configEntry, int materialIndex, vsg::vec4 def);
};
}
#endif
