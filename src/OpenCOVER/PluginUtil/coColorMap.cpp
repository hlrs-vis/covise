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
    //colorMapEntries["Editable"];
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

covise::ColorMapSelector::ColorMapSelector(opencover::ui::Menu& menu)
    : m_selector(new opencover::ui::SelectionList{ &menu, "mapChoice" })
, m_colors(readColorMaps())
{
    for (auto &n: m_colors)
        m_selector->append(n.first);
    m_selector->select(0);
    m_selectedMap = m_colors.begin();

    m_selector->setCallback([this](int index) {
        updateSelectedMap();
        });
}

bool covise::ColorMapSelector::setValue(const std::string& colorMapName)
{
    auto it = m_colors.find(colorMapName);
    if (it == m_colors.end())
        return false;

    m_selector->select(std::distance(m_colors.begin(), it));
    updateSelectedMap();
    return true;
}

osg::Vec4 covise::ColorMapSelector::getColor(float val, float min, float max)
{
    return covise::getColor(val, m_selectedMap->second, min, max);
}

const covise::ColorMap& covise::ColorMapSelector::selectedMap() const
{
    return m_selectedMap->second;
}

void covise::ColorMapSelector::updateSelectedMap()
{
    m_selectedMap = m_colors.begin();
    std::advance(m_selectedMap, m_selector->selectedIndex());
    assert(m_selectedMap != m_colors.end());
}
