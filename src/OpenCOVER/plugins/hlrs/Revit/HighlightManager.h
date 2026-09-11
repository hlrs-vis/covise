#pragma once
#include <osg/Group>
#include <map>
#include <array>
#include <vector>
#include <span>

namespace opencover {
    class coVRSelectionManager;
}
namespace Revit {
struct Color {
    float red = 0.0f, green = 0.0f, blue = 0.0f;
    bool operator==(const Color&) const = default;
};
using ColorIndex = size_t;

class HighlightManager final {
public:
    HighlightManager(opencover::coVRSelectionManager& manager);
    void add(std::span<osg::ref_ptr<osg::Node>> nodes, Color color);
    void remove(osg::ref_ptr<osg::Node> node);
    void highlight(osg::ref_ptr<osg::Node> node, bool enable);
    bool isHighlightedNode(osg::ref_ptr<osg::Node> node);

private:
    size_t findColorIdx(const Color& color);

    std::map<osg::ref_ptr<osg::Node>, ColorIndex> _highlightedNodes;
    std::vector<Color> _palette;
    opencover::coVRSelectionManager* _selectionManager;
};
}