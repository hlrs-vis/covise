#ifndef COVER_PLUGIN_TOOLCHANGER_TOOL_H
#define COVER_PLUGIN_TOOLCHANGER_TOOL_H

#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osgAnimation/StackedScaleElement>
#include <osgAnimation/StackedTransform>

#include <memory>
class ToolModel
{
public:
    typedef std::unique_ptr<ToolModel> ptr;
    ToolModel(osg::Node *model);
    void resize(float length, float radius);
    float getLength() const;
    float getRadius() const;
    void setParent(osg::Group *parent);
    osg::Group* parent();
    osg::MatrixTransform *model() { return m_tool; }
private:
    osg::ref_ptr<osg::MatrixTransform> m_shaft, m_tip, m_tool;
    osgAnimation::StackedScaleElement *m_shaftScale, *m_tipScale;

};


#endif // COVER_PLUGIN_TOOLCHANGER_TOOL_H