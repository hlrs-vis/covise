#include "ColorMapShader.h"
#include <PluginUtil/coShaderUtil.h>

using namespace opencover;

coVRShader *applyPointShader(osg::Drawable *drawable, const ColorMap &colorMap)
{
    return applyShader(drawable, colorMap, "OctPoints");

}

coVRShader *applySurfaceShader(osg::Drawable *drawable, const ColorMap &colorMap)
{
    return applyShader(drawable, colorMap, "MapColorsAttrib");
}