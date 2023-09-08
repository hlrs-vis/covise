#include "ColorMapShader.h"
#include <PluginUtil/coShaderUtil.h>

using namespace opencover;

coVRShader *applyPointShader(osg::Drawable *drawable, const covise::ColorMap &colorMap, float min, float max)
{
    return applyShader(drawable, colorMap, min, max, "OctPoints");

}

coVRShader *applySurfaceShader(osg::Drawable *drawable, const covise::ColorMap &colorMap, float min, float max)
{
    return applyShader(drawable, colorMap, min, max, "MapColorsAttrib");
}