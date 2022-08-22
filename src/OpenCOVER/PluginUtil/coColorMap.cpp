#include "coColorMap.h"
#include <cassert>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <iostream>

using namespace std;
covise::ColorMaps covise::readColorMaps()
{
    // read the name of all colormaps in file

    covise::coCoviseConfig::ScopeEntries colorMapEntries = coCoviseConfig::getScopeEntries("Colormaps");
    ColorMaps colorMaps;
#ifdef NO_COLORMAP_PARAM
    colorMapEntries["COVISE"];
#else
    colorMapEntries["Editable"];
#endif

    for (const auto &map : colorMapEntries)
    {
        string name = "Colormaps." + map.first;

        auto no = coCoviseConfig::getScopeEntries(name).size();
        ColorMap &colorMap = colorMaps.emplace(map.first, ColorMap()).first->second;
        // read all sampling points
        float diff = 1.0f / (no - 1);
        float pos = 0;
        for (int j = 0; j < no; j++)
        {
            string tmp = name + ".Point:" + std::to_string(j);
            ColorMap cm;
            colorMap.r.push_back(coCoviseConfig::getFloat("r", tmp, 0));
            colorMap.g.push_back(coCoviseConfig::getFloat("g", tmp, 0));
            colorMap.b.push_back(coCoviseConfig::getFloat("b", tmp, 0));
            colorMap.a.push_back(coCoviseConfig::getFloat("a", tmp, 1));
            colorMap.samplingPoints.push_back(coCoviseConfig::getFloat("x", tmp, pos));
            pos += diff;
        }
    }
    return colorMaps;
}

osg::Vec4 covise::getColor(float val, const covise::ColorMap& colorMap, float min, float max)
{
    assert(val >= min && val <= max);
    val = 1 / (max - min) * (val - min);

    size_t idx = 0;
    for (; idx < colorMap.samplingPoints.size() && colorMap.samplingPoints[idx + 1] < val; idx++){}


    double d = (val - colorMap.samplingPoints[idx]) / (colorMap.samplingPoints[idx + 1] - colorMap.samplingPoints[idx]);
    osg::Vec4 color;
    color[0] = ((1 - d) * colorMap.r[idx] + d * colorMap.r[idx + 1]);
    color[1] = ((1 - d) * colorMap.g[idx] + d * colorMap.g[idx + 1]);
    color[2] = ((1 - d) * colorMap.b[idx] + d * colorMap.b[idx + 1]);
    color[3] = ((1 - d) * colorMap.a[idx] + d * colorMap.a[idx + 1]);

    return color;
}