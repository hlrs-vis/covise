#ifndef COVER_TOOLMACHINE_CURRENTS_H
#define COVER_TOOLMACHINE_CURRENTS_H

#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Array>
#include <cover/ui/Group.h>
#include <cover/ui/Slider.h>
#include <memory>

class SelfDeletingCurrents;

class Currents 
{
public:
    friend SelfDeletingCurrents;
    Currents(opencover::ui::Group* group, const osg::Node *toolHeadNode, const osg::Node *tableNode);
    void update();
    void setOffset(const std::array<double, 5> &offsets);
        
private:
bool init();
osg::ref_ptr<osg::MatrixTransform> m_generalOffset, m_tableProxy, m_cAxis;
osg::ref_ptr<osg::Geometry> m_traceLine;
osg::ref_ptr<osg::Vec3Array> m_points;
osg::ref_ptr<osg::DrawArrays> m_drawArrays;
const osg::Node *m_toolHeadNode = nullptr;
const osg::Node *m_tableNode = nullptr;
std::unique_ptr<opencover::ui::Group> m_group;
opencover::ui::Slider *m_numPointsSlider;
bool m_clear = false;
};

class SelfDeletingCurrents : public osg::Observer
{
public:
    typedef std::map<std::string, std::unique_ptr<SelfDeletingCurrents>> Map;
    SelfDeletingCurrents(Map &currentsMap, const std::string &name, std::unique_ptr<Currents> &&currents);
    void objectDeleted(void*) override;
    std::unique_ptr<Currents> value;
private:
    Map::iterator m_iter;
    Map &m_currents;
};



#endif // COVER_TOOLMACHINE_CURRENTS_H
