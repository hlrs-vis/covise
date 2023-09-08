
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
    osg::Vec4 PLUGIN_UTILEXPORT getColor(float val, const ColorMap &colorMap, float min = 0, float max = 1);
    //same logic as colors module, but sets linear sampling points
    ColorMap PLUGIN_UTILEXPORT interpolateColorMap(const ColorMap &cm, int numSteps);
    ColorMap PLUGIN_UTILEXPORT upscale(const ColorMap &baseMap, size_t numSteps);

    class PLUGIN_UTILEXPORT ColorMapSelector
    {
    public:
        ColorMapSelector(opencover::ui::Menu &menu);
        ColorMapSelector(opencover::ui::Group &menu);

        bool setValue(const std::string &colorMapName);
        osg::Vec4 getColor(float val, float min = 0, float max = 1);
        const ColorMap &selectedMap() const;
        void setCallback(const std::function<void(const ColorMap &)> &f);

    private:
        opencover::ui::SelectionList *m_selector;
        const ColorMaps m_colors;
        ColorMaps::const_iterator m_selectedMap;
        void updateSelectedMap();
        void init();
};
}

#endif
