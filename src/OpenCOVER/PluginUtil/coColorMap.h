
#ifndef COVUSE_UTIL_COMORMAP_H
#define COVUSE_UTIL_COMORMAP_H

#include <string>
#include <vector>
#include <map>
#include <osg/Vec4>

#include "util/coExport.h"

#include <cover/ui/SelectionList.h>
#include <cover/ui/Menu.h>

namespace covise{

    struct ColorMap
    {
        std::vector<float> r, g, b, a, samplingPoints;
    };

    typedef std::map<std::string, ColorMap> ColorMaps;
    PLUGIN_UTILEXPORT ColorMaps readColorMaps();
    osg::Vec4 PLUGIN_UTILEXPORT getColor(float val, const covise::ColorMap &colorMap, float min = 0, float max = 1);
    

class PLUGIN_UTILEXPORT ColorMapSelector {
public:
    ColorMapSelector(opencover::ui::Menu &menu);
    bool setValue(const std::string &colorMapName);
    osg::Vec4 getColor(float val, float min = 0, float max = 1);
    const ColorMap& selectedMap() const;
private:
    opencover::ui::SelectionList m_selector;
    const ColorMaps m_colors;
    ColorMaps::const_iterator m_selectedMap;
    void updateSelectedMap();
};
}

#endif
