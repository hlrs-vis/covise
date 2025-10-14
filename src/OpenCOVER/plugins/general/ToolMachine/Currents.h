#ifndef COVER_TOOLMACHINE_CURRENTS_H
#define COVER_TOOLMACHINE_CURRENTS_H

#include "Tool.h"

#include <osg/Array>
#include <osg/Geometry>


class Currents : public Tool
{
public:
    Currents(opencover::ui::Group *group, opencover::config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
private:
    void clear() override;
    void applyShader(const opencover::ColorMap& map) override;
    std::vector<std::string> getAttributes() override;
    void initGeo();
    void updateGeo(bool paused, const opencover::dataclient::MultiDimensionalArray<double> &data) override;
    void attributeChanged(float value) override;

    osg::ref_ptr<osg::Geometry> m_traceLine;
    osg::ref_ptr<osg::Vec3Array> m_vertices;
    osg::ref_ptr<osg::FloatArray> m_values;
    osg::ref_ptr<osg::DrawArrays> m_drawArrays;

};


#endif // COVER_TOOLMACHINE_CURRENTS_H
