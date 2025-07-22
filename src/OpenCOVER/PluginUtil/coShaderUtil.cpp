#include "coShaderUtil.h"
#include <osg/Texture1D>
#include "osg/Geode"

using namespace opencover;

constexpr int TfTexUnit = 1;


osg::ref_ptr<osg::Texture1D> opencover::colorMapTexture(const ColorMap &colorMap)
{
    osg::ref_ptr<osg::Texture1D> texture = new osg::Texture1D{};
    texture->setInternalFormat(GL_RGBA8);

    texture->setBorderWidth(0);
    texture->setResizeNonPowerOfTwoHint(false);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    texture->setWrap(osg::Texture1D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);

    osg::ref_ptr<osg::Image> image(new osg::Image);
    image->allocateImage(colorMap.steps(), 1, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    unsigned char *rgba = image->data();
    for (size_t i = 0; i < colorMap.steps(); ++i)
    {
        auto color = colorMap.getColorPerStep(i);
        rgba[4 * i + 0] = 255 * color.r();
        rgba[4 * i + 1] = 255 * color.g();
        rgba[4 * i + 2] = 255 * color.b();
        rgba[4 * i + 3] = 255 * color.a();
    }

    texture->setImage(image);

    return texture;
}

coVRShader *opencover::applyShader(osg::Drawable *drawable, const ColorMap &colorMap, const std::string& shaderFile)
{
    std::map<std::string, std::string> parammap;
    parammap["dataAttrib"] = std::to_string(DataAttrib);
    parammap["texUnit1"] = std::to_string(TfTexUnit);

    auto shader = opencover::coVRShaderList::instance()->getUnique(shaderFile, &parammap);

    osg::ref_ptr<osg::Texture1D> texture = colorMapTexture(colorMap);

    auto state = drawable->getOrCreateStateSet();
    state->setTextureAttribute(TfTexUnit, texture, osg::StateAttribute::ON);
    shader->setFloatUniform("rangeMin", colorMap.min());
    shader->setFloatUniform("rangeMax", colorMap.max());
    shader->apply(state);
    drawable->setStateSet(state);
    return shader;
}

coVRShader *opencover::applyShader(osg::Geode *geo, const ColorMap &colorMap, const std::string& shaderFile)
{
    std::map<std::string, std::string> parammap;
    parammap["dataAttrib"] = std::to_string(DataAttrib);
    parammap["texUnit1"] = std::to_string(TfTexUnit);

    auto shader = opencover::coVRShaderList::instance()->getUnique(shaderFile, &parammap);

    osg::ref_ptr<osg::Texture1D> texture = colorMapTexture(colorMap);

    auto state = geo->getOrCreateStateSet();
    state->setTextureAttribute(TfTexUnit, texture, osg::StateAttribute::ON);
    shader->setFloatUniform("rangeMin", colorMap.min());
    shader->setFloatUniform("rangeMax", colorMap.max());
    shader->apply(state);
    geo->setStateSet(state);
    return shader;
}
