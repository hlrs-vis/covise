#ifndef LAMURE_EDIT_TOOL_H
#define LAMURE_EDIT_TOOL_H

#include <cstdint>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <osg/BoundingBox>
#include <osg/Matrix>
#include <osg/Quat>
#include <osg/Vec3>
#include <lamure/types.h>

class Lamure;
namespace opencover { class coVR3DTransInteractor; }

class LamureEditTool {
public:
    enum class BrushAction { None, Erase, Restore };

    struct NodeHit {
        lamure::model_t model_id{lamure::invalid_model_t};
        lamure::node_t node_id{lamure::invalid_node_t};
        uint16_t model_index{0};
    };

    struct SurfelKey {
        lamure::model_t model{lamure::invalid_model_t};
        lamure::node_t node{lamure::invalid_node_t};
        uint32_t surfel{0};
        bool operator==(const SurfelKey& rhs) const noexcept {
            return model == rhs.model && node == rhs.node && surfel == rhs.surfel;
        }
    };

    struct SurfelKeyHash {
        std::size_t operator()(const SurfelKey& k) const noexcept {
            std::size_t h1 = std::hash<lamure::model_t>{}(k.model);
            std::size_t h2 = std::hash<lamure::node_t>{}(k.node);
            std::size_t h3 = std::hash<uint32_t>{}(k.surfel);
            return ((h1 * 1315423911u) ^ (h2 << 11)) ^ h3;
        }
    };

    struct NodeKey {
        lamure::model_t model{lamure::invalid_model_t};
        lamure::node_t node{lamure::invalid_node_t};
        bool operator==(const NodeKey& rhs) const noexcept {
            return model == rhs.model && node == rhs.node;
        }
    };

    struct NodeKeyHash {
        std::size_t operator()(const NodeKey& k) const noexcept {
            std::size_t h1 = std::hash<lamure::model_t>{}(k.model);
            std::size_t h2 = std::hash<lamure::node_t>{}(k.node);
            return (h1 * 1315423911u) ^ (h2 << 11);
        }
    };

    explicit LamureEditTool(Lamure* plugin);
    ~LamureEditTool();

    void setBrushPose(const osg::Vec3& center, const osg::Vec3& scale);
    void setBrushAction(BrushAction action);
    void clearBrush();
    bool hasBrush() const noexcept { return m_brush.active; }
    BrushAction brushAction() const noexcept { return m_action; }

    std::vector<NodeHit> collectNodesInBrush(bool requireVisible) const;
    std::vector<SurfelKey> collectSurfelsInBrush(BrushAction action, bool requireVisible) const;
    std::size_t applyBrush(BrushAction action, bool requireVisible);
    void markErased(const SurfelKey& key);
    void markRestored(const SurfelKey& key);
    bool isErased(const SurfelKey& key) const;
    bool hasNodeEdits(lamure::model_t model, lamure::node_t node) const;
    void clearEdits();
    std::size_t erasedCount() const noexcept { return m_erased.size(); }
    const std::vector<uint32_t>* erasedInNode(lamure::model_t model, lamure::node_t node) const;
    void updateBrushInteraction(opencover::coVR3DTransInteractor* interactor);
    void enable();
    void disable(bool resetPose = true, bool maskBrush = true);
    void update();
    bool isEnabled() const noexcept { return m_enabled; }
    void applySizeChange(float newSize);
    float brushSize() const noexcept { return m_brushSize; }
    osg::Vec3 brushCenter() const noexcept { return m_brush.center; }
    void updateSizeInteractorPose(const osg::Vec3& center);

private:
    struct BrushState {
        osg::Vec3 center{0.f, 0.f, 0.f};
        osg::Vec3 scale{1.f, 1.f, 1.f};
        bool active{false};
    };

    bool containsSurfel(const osg::Vec3& worldPos, float surfelRadius) const;
    void applyBrushMask();

    Lamure* m_plugin{nullptr};
    BrushState m_brush;
    BrushAction m_action{BrushAction::None};
    std::unordered_set<SurfelKey, SurfelKeyHash> m_erased;
    std::unordered_map<NodeKey, std::vector<uint32_t>, NodeKeyHash> m_erased_by_node;
    opencover::coVR3DTransInteractor* m_planar_interactor{nullptr};
    opencover::coVR3DTransInteractor* m_depth_interactor{nullptr};
    opencover::coVR3DTransInteractor* m_size_interactor{nullptr};
    bool m_enabled{false};
    osg::Vec3 m_initialDragOffsetView{0.f, 3.f, 0.f}; // default view offset in front of camera
    osg::Vec3 m_dragOffsetView;   // user drag offset in view space
    osg::Vec3 m_depthOffsetView{0.f, 0.f, 0.f}; // lateral offset for depth interactor in view space
    osg::Vec3 m_sizeOffsetView{0.f, 0.f, 0.f};  // lateral offset for size interactor in view space
    osg::Matrixd m_brushLocalMat; // current brush matrix (local)
    float m_brushSize{1.f};       // current brush size (radius)
    float m_interactorBaseSize{1.f}; // constant visual size of interactors
    bool m_brushMasked{false};
    bool m_savedPoseValid{false};
    osg::Vec3 m_savedBrushWorldPos{0.f, 0.f, 0.f};
};

#endif // LAMURE_EDIT_TOOL_H
