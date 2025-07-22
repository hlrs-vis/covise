#ifndef COVISE_COLOR_MAP_SHADER_H
#define COVISE_COLOR_MAP_SHADER_H
#include <PluginUtil/colors/coColorMap.h>
#include <cover/coVRShader.h>
    
constexpr int TfTexUnit = 1;
constexpr int DataAttrib = 10;
opencover::coVRShader *applyPointShader(osg::Drawable *drawable, const opencover::ColorMap &colorMap);
opencover::coVRShader *applySurfaceShader(osg::Drawable *drawable, const opencover::ColorMap &colorMap);

#endif // COVISE_COLOR_MAP_SHADER_H