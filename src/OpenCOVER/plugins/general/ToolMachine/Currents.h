#ifndef COVER_TOOLMACHINE_CURRENTS_H
#define COVER_TOOLMACHINE_CURRENTS_H

#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Array>
#include <cover/ui/Group.h>
#include <cover/ui/Slider.h>
#include <memory>
#include <PluginUtil/coColorMap.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/EditField.h>
class SelfDeletingCurrents;

namespace opencover{namespace opcua{
class Client;
}}

class Currents 
{
public:
    friend SelfDeletingCurrents;
    Currents(opencover::ui::Group* group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
    ~Currents();
    void update();
        
private:
void initGeo();
osg::ref_ptr<osg::Geometry> m_traceLine;
osg::ref_ptr<osg::Vec3Array> m_points;
osg::ref_ptr<osg::FloatArray> m_values;
osg::ref_ptr<osg::DrawArrays> m_drawArrays;
osg::MatrixTransform *m_toolHeadNode = nullptr;
osg::MatrixTransform *m_tableNode = nullptr;
std::unique_ptr<opencover::ui::Group> m_group;
opencover::ui::Slider *m_numPointsSlider;
bool m_clear = false;
covise::ColorMapSelector *m_colorMapSelector;
opencover::ui::SelectionList *m_attributeName;
opencover::ui::EditField *m_minAttribute, *m_maxAttribute;
opencover::opcua::Client *m_client;
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
