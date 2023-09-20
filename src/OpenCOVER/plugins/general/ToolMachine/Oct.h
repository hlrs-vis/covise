#ifndef COVER_TOOLMACHINE__H
#define COVER_TOOLMACHINE__H

#include "Tool.h"

#include <osg/Array>
#include <osg/Geometry>
#include <cover/ui/Button.h>

class Oct : public Tool
{
public:
    Oct(opencover::ui::Group *group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
    void setOffset(const std::string &name);    
    void setScale(float scale);
private:
    struct Section
    {
        enum Visibility{
            Invisible = 0,
            PointsOnly = (1u << 0),
            SurfaceOnly = (1u << 1),
            Visible = PointsOnly | SurfaceOnly};

        Section(double pointSize, const covise::ColorMap& map, float min, float max, osg::MatrixTransform* parent);
        ~Section();
        bool append(const osg::Vec3 &pos, float spiecies);
        void createSurface();
        void show(Visibility status = Visible);
        osg::ref_ptr<osg::Geometry> points, surface;
        osg::ref_ptr<osg::Vec3Array> vertices;
        osg::ref_ptr<osg::FloatArray> species;
        osg::ref_ptr<osg::DrawArrays> pointPrimitiveSet;
        osg::ref_ptr<osg::DrawElementsUInt> surfacePrimitiveSet, reducedPointPrimitiveSet;
        bool changed = true;
        Visibility status = Visible;
    private:
        osg::MatrixTransform* m_parent;
    };
    void clear() override;
    void applyShader(const covise::ColorMap& map, float min, float max) override;
    std::vector<std::string> getAttributes() override;
    void initGeo();
    void updateGeo(bool paused) override;
    Section &addSection();
    void addPoints(const std::string &valueName, const osg::Vec3 &toolHeadPos, const osg::Vec3 &up, float radius);
    void correctLastUpdate(const osg::Vec3 &toolHeadPos);
    struct Update{
        size_t numValues;
        osg::Vec3 pos;
    } m_lastUpdate;

    std::deque<Section> m_sections, m_reducedSections;
    opencover::ui::Slider *m_pointSizeSlider;
    opencover::ui::Button *m_showSurfaceBtn;
    std::string m_offsetName;
    opencover::opcua::ObserverHandle m_opcuaOffsetId;
    float m_opcUaToVrmlScale = 1;


};


#endif // COVER_TOOLMACHINE__H
