#pragma once
#include <osg/Group>
#include <map>
#include <vector>
#include <span>
#include <functional>

namespace opencover
{
class coVRSelectionManager;
}
namespace Revit
{
struct Color
{
    float red = 0.0f, green = 0.0f, blue = 0.0f;
    bool operator==(const Color &) const = default;
};

using ColorIndex = size_t;
using Node = osg::ref_ptr<osg::Node>;

class HighlightManager final
{
public:
    using HighlightStrategy = std::function<void(Node, Color const &, bool)>;

    HighlightManager(HighlightStrategy highlighter)
        : _highlighter { highlighter } { };
    void add(std::span<Node> nodes, Color color);
    void remove(Node node);
    void highlight(Node node, bool enable);
    bool isHighlightedNode(Node node);
    void setHighlighter(HighlightStrategy higlighter);

private:
    size_t findColorIdx(const Color &color);

    std::map<Node, ColorIndex> _highlightedNodes;
    std::vector<Color> _palette;
    HighlightStrategy _highlighter;
};
}
