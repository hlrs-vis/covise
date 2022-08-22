
#ifndef COVUSE_UTIL_COMORMAP_H
#define COVUSE_UTIL_COMORMAP_H

#include <string>
#include <vector>
#include <map>
#include <osg/Vec4>
#include "util/coExport.h"

namespace covise{

    struct ColorMap
    {
        std::vector<float> r, g, b, a, samplingPoints;
    };

    typedef std::map<std::string, ColorMap> ColorMaps;
    PLUGIN_UTILEXPORT ColorMaps readColorMaps();
    osg::Vec4 PLUGIN_UTILEXPORT getColor(float val, const covise::ColorMap &colorMap, float min = 0, float max = 1);
    
}

#endif