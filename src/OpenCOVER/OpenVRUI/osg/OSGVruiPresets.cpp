/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiPresets.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiLog.h>
#include <config/CoviseConfig.h>

#include <osg/AlphaFunc>

using namespace osg;

namespace vrui
{

using covise::coCoviseConfig;

OSGVruiPresets *OSGVruiPresets::instance = 0;

OSGVruiPresets::OSGVruiPresets()
{

    polyModeFill = new PolygonMode();
    polyModeFill->setMode(PolygonMode::FRONT_AND_BACK, PolygonMode::FILL);

    cullFaceBack = new CullFace();
    cullFaceBack->setMode(CullFace::BACK);

    oneMinusSourceAlphaBlendFunc = new BlendFunc();
    oneMinusSourceAlphaBlendFunc->setSource(GL_SRC_ALPHA);
    oneMinusSourceAlphaBlendFunc->setDestination(GL_ONE_MINUS_SRC_ALPHA);

    texEnvModulate = new TexEnv();
    texEnvModulate->setMode(TexEnv::MODULATE);

    for (int i = 0; i < coUIElement::NUM_MATERIALS; ++i)
    {
        ref_ptr<Material> material = new Material();

        material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        material->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        material->setShininess(Material::FRONT_AND_BACK, 0.0f);

        materials.push_back(material);
    }

    materials[coUIElement::RED]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.0f, 0.0f, 1.0f));
    materials[coUIElement::RED]->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    materials[coUIElement::GREEN]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0f, 0.2f, 0.0f, 1.0f));
    materials[coUIElement::GREEN]->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.0f, 1.0f, 0.0f, 1.0f));

    materials[coUIElement::BLUE]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.2f, 1.0f));
    materials[coUIElement::BLUE]->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 1.0f, 1.0f));

    materials[coUIElement::YELLOW]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.0f, 1.0f));
    materials[coUIElement::YELLOW]->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 0.0f, 1.0f));

    materials[coUIElement::GREY]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.4f, 0.4f, 0.4f, 1.0f));
    materials[coUIElement::GREY]->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.1f, 0.1f, 0.1f, 1.0f));

    materials[coUIElement::WHITE]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.3f, 0.3f, 0.3f, 1.0f));
    materials[coUIElement::WHITE]->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));

    materials[coUIElement::BLACK]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    materials[coUIElement::BLACK]->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));

    materials[coUIElement::DARK_YELLOW]->setAmbient(Material::FRONT_AND_BACK, Vec4(0.3f, 0.5f, 0.0f, 1.0f));
    materials[coUIElement::DARK_YELLOW]->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.3f, 0.5f, 0.0f, 1.0f));

    materials[coUIElement::WHITE_NL]->setAmbient(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    materials[coUIElement::WHITE_NL]->setEmission(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    materials[coUIElement::WHITE_NL]->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));

    setColorFromConfig("COVER.VRUI.ItemBackgroundNormal", coUIElement::ITEM_BACKGROUND_NORMAL, Vec4(0.1f, 0.1f, 0.1f, 1.0f));
    setColorFromConfig("COVER.VRUI.ItemBackgroundHighlighted", coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, Vec4(0.3f, 0.5f, 0.0f, 1.0f));
    setColorFromConfig("COVER.VRUI.ItemBackgroundDisabled", coUIElement::ITEM_BACKGROUND_DISABLED, Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    setColorFromConfig("COVER.VRUI.HandleBackgroundNormal", coUIElement::HANDLE_BACKGROUND_NORMAL, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    setColorFromConfig("COVER.VRUI.HandleBackgroundHighlighted", coUIElement::HANDLE_BACKGROUND_HIGHLIGHTED, Vec4(0.6f, 0.6f, 0.0f, 1.0f));
    setColorFromConfig("COVER.VRUI.HandleBackgroundDisabled", coUIElement::HANDLE_BACKGROUND_DISABLED, Vec4(0.6f, 0.6f, 0.6f, 1.0f));

    for (int i = 0; i < coUIElement::NUM_MATERIALS; ++i)
    {

        ref_ptr<StateSet> stateSet = new StateSet();

        stateSet->setGlobalDefaults();
        stateSet->setAttributeAndModes(materials[i].get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setAttributeAndModes(polyModeFill.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setMode(GL_CULL_FACE, StateAttribute::OFF | StateAttribute::PROTECTED);
        stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setMode(GL_BLEND, StateAttribute::OFF | StateAttribute::PROTECTED);

        ref_ptr<StateSet> stateSetCulled = new StateSet();
        stateSetCulled->setGlobalDefaults();
        stateSetCulled->setAttributeAndModes(materials[i].get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSetCulled->setAttributeAndModes(polyModeFill.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSetCulled->setAttributeAndModes(cullFaceBack.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSetCulled->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
        stateSetCulled->setMode(GL_BLEND, StateAttribute::OFF | StateAttribute::PROTECTED);
        if ((i == coUIElement::WHITE_NL) || (i == coUIElement::ITEM_BACKGROUND_NORMAL) || (i == coUIElement::ITEM_BACKGROUND_HIGHLIGHTED) || (i == coUIElement::ITEM_BACKGROUND_DISABLED) || (i == coUIElement::HANDLE_BACKGROUND_NORMAL) || (i == coUIElement::HANDLE_BACKGROUND_HIGHLIGHTED) || (i == coUIElement::HANDLE_BACKGROUND_DISABLED))
        {
            stateSet->setMode(GL_LIGHTING, StateAttribute::OFF | StateAttribute::PROTECTED);
            stateSetCulled->setMode(GL_LIGHTING, StateAttribute::OFF | StateAttribute::PROTECTED);
        }

        stateSets.push_back(stateSet);
        stateSetsCulled.push_back(stateSetCulled);

        fontFile = "share/covise/fonts/" + coCoviseConfig::getEntry("value", "COVER.VRUI.Font", coCoviseConfig::getEntry("value", "COVER.Font", "DroidSansFallbackFull.ttf"));
        fontFile = vruiRendererInterface::the()->getName(fontFile);
    }
}

OSGVruiPresets::~OSGVruiPresets()
{
}

StateSet *OSGVruiPresets::getStateSet(coUIElement::Material material)
{

    //static CopyOp copyOp(CopyOp::DEEP_COPY_STATESETS);

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("OSGVruiPresets::getStateSet err: material " << material << " not found");
        return 0;
    }
    if (instance == 0)
        instance = new OSGVruiPresets();

    //return dynamic_cast<StateSet*>(instance->stateSets[material]->clone(copyOp));
    return instance->stateSets[material].get();
}

StateSet *OSGVruiPresets::getStateSetCulled(coUIElement::Material material)
{

    //   static CopyOp copyOp(CopyOp::DEEP_COPY_STATESETS);

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("OSGVruiPresets::getStateSetCulled err: material " << material << " not found");
        return 0;
    }
    if (instance == 0)
        instance = new OSGVruiPresets();

    //  return dynamic_cast<StateSet*>(instance->stateSetsCulled[material]->clone(copyOp));
    return instance->stateSetsCulled[material].get();
}

StateSet *OSGVruiPresets::makeStateSet(coUIElement::Material material)
{

    static CopyOp copyOp(CopyOp::DEEP_COPY_STATESETS);

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("OSGVruiPresets::makeStateSet err: material " << material << " not found");
        return 0;
    }
    if (instance == 0)
        instance = new OSGVruiPresets();

    return dynamic_cast<StateSet *>(instance->stateSets[material]->clone(copyOp));
}

StateSet *OSGVruiPresets::makeStateSetCulled(coUIElement::Material material)
{

    static CopyOp copyOp(CopyOp::DEEP_COPY_STATESETS);

    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("OSGVruiPresets::makeStateSetCulled err: material " << material << " not found");
        return 0;
    }
    if (instance == 0)
        instance = new OSGVruiPresets();

    return dynamic_cast<StateSet *>(instance->stateSetsCulled[material]->clone(copyOp));
}

osg::Material *OSGVruiPresets::getMaterial(coUIElement::Material material)
{
    if (material >= coUIElement::NUM_MATERIALS)
    {
        VRUILOG("OSGVruiPresets::getMaterial err: material " << material << " not found");
        return 0;
    }
    if (instance == 0)
        instance = new OSGVruiPresets();
    return instance->materials[material].get();
}

TexEnv *OSGVruiPresets::getTexEnvModulate()
{
    if (instance == 0)
        instance = new OSGVruiPresets();
    return instance->texEnvModulate.get();
}

PolygonMode *OSGVruiPresets::getPolyModeFill()
{
    if (instance == 0)
        instance = new OSGVruiPresets();
    return instance->polyModeFill.get();
}

CullFace *OSGVruiPresets::getCullFaceBack()
{
    if (instance == 0)
        instance = new OSGVruiPresets();
    return instance->cullFaceBack.get();
}

BlendFunc *OSGVruiPresets::getBlendOneMinusSrcAlpha()
{
    if (instance == 0)
        instance = new OSGVruiPresets();
    return instance->oneMinusSourceAlphaBlendFunc.get();
}

void OSGVruiPresets::makeTransparent(osg::StateSet *state, bool continuous)
{
    static bool useAlphaTest = true;

    if (useAlphaTest)
    {
        AlphaFunc *alphaFunc = new AlphaFunc(AlphaFunc::GREATER, 0.0);
        state->setAttributeAndModes(alphaFunc, StateAttribute::ON);
    }

    if (continuous)
    {
        state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        state->setNestRenderBins(false);
    }
}

std::string OSGVruiPresets::getFontFile()
{
    if (instance == 0)
        instance = new OSGVruiPresets();
    return instance->fontFile;
}

void OSGVruiPresets::setColorFromConfig(const char *configEntry, int materialIndex, osg::Vec4 def)
{
    osg::Vec4 color;
    color = osg::Vec4(coCoviseConfig::getFloat("r", configEntry, def[0]),
                      coCoviseConfig::getFloat("g", configEntry, def[1]),
                      coCoviseConfig::getFloat("b", configEntry, def[2]), 1.0f);
    materials[materialIndex]->setAmbient(Material::FRONT_AND_BACK, color);
    materials[materialIndex]->setDiffuse(Material::FRONT_AND_BACK, color);
}
}
