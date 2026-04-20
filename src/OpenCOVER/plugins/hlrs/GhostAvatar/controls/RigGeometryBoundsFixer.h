#include <osg/Geode>
#include <osg/NodeVisitor>

/*
    Traverses through all children and changes their `ComputeBoundingBoxCallback`
    to `RigGeometryComputeBoundingBoxCallback`, if they are of type `RigGeometry`.
    This is necessary, because the default callback seems to calculate the wrong
    bounding box for `RigGeometry` nodes.
*/
struct RigGeometryBoundsFixer : public osg::NodeVisitor
{
    RigGeometryBoundsFixer()
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
    {
    }

    void apply(osg::Geode &geode) override;
};