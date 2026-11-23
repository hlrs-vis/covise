#pragma once

#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>

// ---------------------------------------------------------------------------
// ClipBoxNode: a movable, axis-aligned box in the scene. The geometry is a
// unit cube ([-0.5, 0.5]^3) rendered as a ghost (edges + translucent faces);
// the box's world AABB is always derived from its world transform, so moving
// it with the Move plugin just works.
// ---------------------------------------------------------------------------
class ClipBoxNode : public osg::MatrixTransform
{
public:
    explicit ClipBoxNode(const std::string &name);

    // highlight the box while it is selected (the selection plugin's
    // selectionChanged listener toggles this on the selected box)
    void setSelected(bool selected);
    bool isSelected() const;
    void setSize(const osg::Vec3 &size);

    // set the box's extent in its parent space with an explicit min/max
    // (center-preserving when the box already has a size)
    void setAABB(const osg::Vec3 &min, const osg::Vec3 &max);
    const osg::Vec3 &size() const;

    // the box's world-space AABB: transform the unit cube's corners through
    // the full parent chain (halted at `haltAt`, usually the objects root)
    void worldAABB(osg::Node *haltAt, osg::Vec3 &min, osg::Vec3 &max) const;

private:
    void buildGeometry();
    void setGhostColor(const osg::Vec3 &rgb, float alpha, float emissionAlpha);
    // matrix = scale(size); the translation is the local origin, so the box
    // occupies [origin, origin+size] in parent space (bottom-anchored);
    // the Move plugin composes its moves on top of the current matrix
    void rebuildMatrix();

    osg::StateSet *m_stateSet = nullptr;
    osg::ref_ptr<osg::Material> m_material;
    bool m_selected = false;
    osg::Vec3 m_size{1.0f, 1.0f, 1.0f};
};
