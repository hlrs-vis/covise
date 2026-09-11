#include "HighlightManager.h"
#include <ranges>
#include <osg/MatrixTransform>
#include <cover/coVRSelectionManager.h>

namespace Revit
{

size_t HighlightManager::findColorIdx(const Color &color)
{
    auto it = std::ranges::find(_palette, color);
    if (it == std::ranges::end(_palette))
    {
        _palette.push_back(color);
        return _palette.size() - 1;
    }
    return it - _palette.begin();
}

void HighlightManager::add(std::span<Node> nodes, Color color)
{
    auto idx = findColorIdx(color);
    for (auto node : nodes)
        _highlightedNodes.insert_or_assign(node, idx);
}

void HighlightManager::remove(Node node)
{
    if (!isHighlightedNode(node))
        return;
    highlight(node, false);
    _highlightedNodes.erase(node);
}

void HighlightManager::highlight(Node node, bool enable)
{
    if (!isHighlightedNode(node))
        return;

    auto colorIdx = _highlightedNodes[node];
    auto color = _palette[colorIdx];

    _highlighter(node, color, enable);
}

bool HighlightManager::isHighlightedNode(Node node)
{
    if (!node)
        return false;
    return _highlightedNodes.contains(node);
}

void HighlightManager::setHighlighter(HighlightStrategy higlighter)
{
    _highlighter = higlighter;
}

}
