#include "ColorMapShader.h"
#include <osg/Texture1D>


opencover::coVRShader *applyShader(osg::Geode *geode, osg::Drawable *drawable, const covise::ColorMap &colorMap, float min, float max, const std::string& shaderFile)
{
    std::map<std::string, std::string> parammap;
    parammap["dataAttrib"] = std::to_string(DataAttrib);
    parammap["texUnit1"] = std::to_string(TfTexUnit);

    auto shader = opencover::coVRShaderList::instance()->getUnique(shaderFile, &parammap);
    

    osg::ref_ptr<osg::Texture1D> texture = new osg::Texture1D{};
    texture->setInternalFormat(GL_RGBA8);

    texture->setBorderWidth(0);
    texture->setResizeNonPowerOfTwoHint(false);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    texture->setWrap(osg::Texture1D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);

    osg::ref_ptr<osg::Image> image(new osg::Image);
    image->allocateImage(colorMap.samplingPoints.size(), 1, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    unsigned char *rgba = image->data();
    for (size_t i = 0; i < colorMap.samplingPoints.size(); ++i)
    {
        rgba[4 * i + 0] = 255 * colorMap.r[i];
        rgba[4 * i + 1] = 255 * colorMap.g[i];
        rgba[4 * i + 2] = 255 * colorMap.b[i];
        rgba[4 * i + 3] = 255 * colorMap.a[i];
    }

    

    texture->setImage(image);

    auto state = drawable->getOrCreateStateSet();
    state->setTextureAttribute(TfTexUnit, texture, osg::StateAttribute::ON);
    shader->setFloatUniform("rangeMin", min);
    shader->setFloatUniform("rangeMax", max);
    shader->apply(state);
    drawable->setStateSet(state);
    return shader;
}

opencover::coVRShader *applyPointShader(osg::Geode *geode, osg::Drawable *drawable, const covise::ColorMap &colorMap, float min, float max)
{
    return applyShader(geode, drawable, colorMap, min, max, "OctPoints");

}
opencover::coVRShader *applySurfaceShader(osg::Geode *geode, osg::Drawable *drawable, const covise::ColorMap &colorMap, float min, float max)
{
    return applyShader(geode, drawable, colorMap, min, max, "MapColorsAttrib");
}