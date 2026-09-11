#include "HighlightManager.h"
#include <ranges>
#include <osg/MatrixTransform>
#include <cover/coVRSelectionManager.h>

namespace Revit {

HighlightManager::HighlightManager(opencover::coVRSelectionManager& manager)
    : _selectionManager(&manager)
{}

size_t HighlightManager::findColorIdx(const Color& color) {
    auto it = std::ranges::find(_palette, color);
    if (it == std::ranges::end(_palette)) {
        _palette.push_back(color);
        return _palette.size() - 1;
    }
    return it - _palette.begin();
}

void HighlightManager::add(std::span<osg::ref_ptr<osg::Node>> nodes, Color color) {
    auto idx = findColorIdx(color);
	for (auto node : nodes)
		_highlightedNodes.insert_or_assign(node, idx);
}

void HighlightManager::remove(osg::ref_ptr<osg::Node> node) {
    if (!isHighlightedNode(node))
        return;
    highlight(node, false);
    _highlightedNodes.erase(node);
}

void HighlightManager::highlight(osg::ref_ptr<osg::Node> node, bool enable) {
    if (!isHighlightedNode(node))
        return;

    if (enable) {
        if (node->getNumParents() == 0)
            return;
        auto colorIdx = _highlightedNodes[node];
        auto color = _palette[colorIdx];
		_selectionManager->setSelectionColor(color.red, color.green, color.blue);
		_selectionManager->addSelection(node->getParent(0), node);
    } else {
        _selectionManager->removeNode(node);
    }
}

bool HighlightManager::isHighlightedNode(osg::ref_ptr<osg::Node> node) {
    if (!node)
        return false;
    return _highlightedNodes.contains(node);
}

}