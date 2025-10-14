#ifndef COVER_TOOLMACHINE__H
#define COVER_TOOLMACHINE__H

#include "Tool.h"

#include <osg/Array>
#include <osg/Geometry>
#include <cover/ui/Button.h>

class Oct : public Tool
{
public:
    Oct(opencover::ui::Group *group, opencover::config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
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

        Section(size_t vertsPerCircle, double pointSize, const opencover::ColorMap& map, osg::MatrixTransform* parent);
        ~Section();
        bool append(const osg::Vec3 &pos, float species);
        void createSurface();
        void show(Visibility status = Visible);
        osg::ref_ptr<osg::Geometry> points, surface;
        osg::ref_ptr<osg::Vec3Array> vertices;
        osg::ref_ptr<osg::FloatArray> species;
        osg::ref_ptr<osg::DrawArrays> pointPrimitiveSet;
        osg::ref_ptr<osg::DrawElementsUInt> surfacePrimitiveSet, reducedPointPrimitiveSet;
        size_t startIndex = 0; //where in the circle the section starts
        bool changed = true;
        Visibility status = Visible;
        size_t vertsPerCircle = 0;

    private:
        osg::MatrixTransform* m_parent;
        bool lastIsOutsidePoint();

    };
    void clear() override;
    void applyShader(const opencover::ColorMap& map) override;
    std::vector<std::string> getAttributes() override;
    void initGeo();
    void updateGeo(bool paused, const opencover::dataclient::MultiDimensionalArray<double> &data) override;
    Section &addSection(size_t numVerts);
    void addPoints(const opencover::dataclient::MultiDimensionalArray<double> &data, const osg::Vec3 &toolHeadPos, const osg::Vec3 &up, float radius);
    void correctLastUpdate(const osg::Vec3 &toolHeadPos);
    void attributeChanged(float value) override;

    osg::Vec3 m_lastUpdatePos;

    std::deque<Section> m_sections;
    opencover::ui::Slider *m_pointSizeSlider;
    opencover::ui::Button *m_showSurfaceBtn;
    opencover::ui::Button *m_switchVecScalar;
    std::string m_offsetName;
    opencover::dataclient::ObserverHandle m_opcuaOffsetId;
    float m_opcUaToVrmlScale = 1;



};


#endif // COVER_TOOLMACHINE__H
