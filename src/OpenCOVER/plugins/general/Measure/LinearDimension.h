#ifndef MEASURE_LINEAR_DIMENSION_H
#define MEASURE_LINEAR_DIMENSION_H
#include "Dimension.h"

#include <cover/ui/VectorEditField.h>
#include <cover/ui/SelectionList.h>


class Presentation
{
public:
    Presentation();
    virtual ~Presentation();
    virtual void update(const osg::Vec3f &pos1, const osg::Vec3f &pos2, const Scaling &scaling) = 0;
protected:
    std::string formatLabelText(float distance);

    osg::ref_ptr<osg::MatrixTransform> m_geo;
};


class LinePresentation : public Presentation
{
public:
    LinePresentation();
    void update(const osg::Vec3f &pos1, const osg::Vec3f &pos2, const Scaling &scaling) override;
private:
    TextBoard m_textLabel; 
};

class ThreeDPresentation : public Presentation
{
public:
    ThreeDPresentation();
    void update(const osg::Vec3f &pos1, const osg::Vec3f &pos2, const Scaling &scaling) override;
private:
    std::array<TextBoard, 3> m_textLabels; 
    std::array<osg::ref_ptr<osg::MatrixTransform>, 3> m_lines; 
};

/** Start and end point of line; represented as cones with spheres on top */
class LinearDimension : public Dimension
{
public:
    LinearDimension(int id, opencover::coVRPlugin *plugin, opencover::ui::Group *parent, const Scaling &scale);
    void update() override;

private:
    void scalingChanged() override;
    osg::ref_ptr<osg::MatrixTransform> line;
    opencover::ui::VectorEditField *m_distanceEdit;
    opencover::ui::SelectionList *m_presentationSelector;
    std::unique_ptr<Presentation> m_presentation;
    float m_oldScale = 0;
    bool m_scalingChanged = false;

};

#endif // MEASURE_LINEAR_DIMENSION_H
