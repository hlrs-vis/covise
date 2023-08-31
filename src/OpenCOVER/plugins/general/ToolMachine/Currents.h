#ifndef COVER_TOOLMACHINE_CURRENTS_H
#define COVER_TOOLMACHINE_CURRENTS_H

#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Array>
class Currents 
{
public:
    Currents();
    void update(const std::array<double, 5> &position, const std::array<double, 5> &currents);
    void setOffset(const std::array<double, 5> &offsets);
private:
osg::ref_ptr<osg::MatrixTransform> m_generalOffset, m_aAxis, m_cAxis;
osg::ref_ptr<osg::Geometry> m_traceLine;
osg::ref_ptr<osg::Vec3Array> m_points;
osg::ref_ptr<osg::DrawArrays> m_drawArrays;

};





#endif // COVER_TOOLMACHINE_CURRENTS_H
