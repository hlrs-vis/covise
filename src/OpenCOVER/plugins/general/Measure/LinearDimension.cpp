#include "LinearDimension.h"
#include <PluginUtil/colors/ColorMaterials.h>
#include "Measure.h"

#include <cover/coVRPluginSupport.h>

#include <osg/ShapeDrawable>

#include <sstream>
using namespace opencover; 


osg::ref_ptr<osg::Geode> createLine(osg::Material *mat)
{
    osg::ref_ptr<osg::Geode> geodeCyl = new osg::Geode;
    geodeCyl->setName("measureLineGeode");
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0, 0, 0.5), 1, 1));
    // sd->setColor(osg::Vec4{1,0,0,1});
    sd->setName("measureLineShape");
    osg::ref_ptr<osg::StateSet> ss = sd->getOrCreateStateSet();
    geodeCyl->addDrawable(sd);
    ss->setAttributeAndModes(mat, osg::StateAttribute::ON);
    return geodeCyl;
}

Presentation::Presentation()
:m_geo(new osg::MatrixTransform)
{
    cover->getObjectsRoot()->addChild(m_geo.get());
}

Presentation::~Presentation()
{
    while (m_geo->getNumParents())
        m_geo->getParent(0)->removeChild(m_geo.get());
}

std::string Presentation::formatLabelText(float distance)
{
    return displayValueWithUnit(distance, cover->getSceneUnit());
}

const osg::Matrix invisibleScale(osg::Matrix::scale(osg::Vec3f{0.001,0.001,0.001}));

LinePresentation::LinePresentation()
{
    m_geo->setMatrix(invisibleScale);
    m_geo->addChild(createLine(material::get(material::White)));
}

auto calcDynamicScaling()
{
    osg::Vec3 wpoint1 = osg::Vec3(0, 0, 0);
    osg::Vec3 wpoint2 = osg::Vec3(0, 0, 300);
    //distance formula
    osg::Vec3 opoint1 = wpoint1 * cover->getInvBaseMat();
    osg::Vec3 opoint2 = wpoint2 * cover->getInvBaseMat();

    //distance formula
    osg::Vec3 wDiff = wpoint2 - wpoint1;
    osg::Vec3 oDiff = opoint2 - opoint1;
    double distWld = wDiff.length();
    double distObj = oDiff.length();
    // Scaling scaling;
    // scaling.geoScale = distObj / ((31 - lineWidth) * distWld);
    // scaling.textScale = distObj / distWld; 
    return distObj / distWld;
}

void LinePresentation::update(const osg::Vec3f &p1, const osg::Vec3f &p2, const Scaling &scaling)
{
        if(p1 == p2)
            return;
        auto vec = p2 - p1;
        float dist = (vec).length();
        osg::Matrix scale, rot, trans;
        scale.makeIdentity();
        rot.makeIdentity();
        trans.makeIdentity();
        float dynamicScale = calcDynamicScaling() * 10;
        auto lineScale = dynamicScale / (31 - scaling.lineWidth());
        scale.makeScale(osg::Vec3d{lineScale,lineScale,dist});
        osg::Vec3f zAxis{0,0,1};
        rot.makeRotate(zAxis, vec);
        trans.setTrans(p1);
        m_geo->setMatrix(scale * rot * trans);

        osg::Matrix textMat;
        auto textScale = dynamicScale * scaling.fontFactor() / 10;
        textMat.makeScale(osg::Vec3f{textScale, textScale, textScale});
        textMat.setTrans(p1 + vec/2 + zAxis * textScale * 2);
        m_textLabel.setPosition(textMat);
        m_textLabel.setText(formatLabelText(dist));
}

ThreeDPresentation::ThreeDPresentation()
:m_lines({new osg::MatrixTransform, new osg::MatrixTransform, new osg::MatrixTransform})
{

    m_geo->setName("ThreeDPresentation");
    m_lines[0]->addChild(createLine(material::get(material::Red)));
    m_lines[1]->addChild(createLine(material::get(material::Green)));
    m_lines[2]->addChild(createLine(material::get(material::Blue)));
    for (size_t i = 0; i < 3; i++)
    {
        m_lines[i]->setName("line_" + std::to_string(i));
        m_geo->addChild(m_lines[i]);
        m_lines[i]->setMatrix(invisibleScale);
    }
    std::cerr << "ThreeDPresentation: num children " << m_geo->getNumChildren() << std::endl;
}

void ThreeDPresentation::update(const osg::Vec3f &pos1, const osg::Vec3f &pos2, const Scaling &scaling)
{
    if(pos2 == pos1)
        return;
    auto vec = pos2 - pos1;
    osg::Vec3f transVec = pos1;
    float dynamicScale = calcDynamicScaling() * 10;
    auto textScale = dynamicScale * scaling.fontFactor() / 10;

    auto lineScale = dynamicScale / (31 - scaling.lineWidth());
    for (size_t i = 0; i < 3; i++)
    {
        osg::Matrix scale, rot, trans;
        osg::Vec3f length{0,0,0};
        length[i] = vec[i];
        scale.makeIdentity();
        rot.makeIdentity();
        trans.makeIdentity();

        scale.makeScale(osg::Vec3d{lineScale,lineScale,std::abs(vec[i])});

        osg::Vec3f zAxis{0,0,1};
        rot.makeRotate(zAxis, length);

        trans.setTrans(transVec);

        m_lines[i]->setMatrix(scale * rot * trans);      

        osg::Matrix textMat;

        textMat.makeScale(osg::Vec3f{textScale, textScale, textScale});
        textMat.setTrans(transVec + length/2 + zAxis * textScale * 2);
        m_textLabels[i].setPosition(textMat);
        m_textLabels[i].setText(formatLabelText(std::abs(vec[i])));

        transVec += length;

    }
}

LinearDimension::LinearDimension(int id, opencover::coVRPlugin *plugin, ui::Group *parent, const Scaling &scale)
: Dimension(id, "distance_" + std::to_string(id), plugin, parent, scale)
, m_distanceEdit(new ui::VectorEditField(m_gui.get(), "distance"))
, m_presentationSelector(new ui::SelectionList(m_gui.get(), "presentation"))
, m_presentation(std::make_unique<LinePresentation>())
{
    addPin();
    addPin();
    m_distanceEdit->setPriority(ui::Element::Priority::Low);
    std::vector<std::string> presentationList{"Line", "3D"};
    m_presentationSelector->setList(presentationList);
    m_presentationSelector->setCallback([this](int index){
        switch (index)
        {
        case 0:
            m_presentation = std::make_unique<LinePresentation>();
            break;
         case 1:
            m_presentation = std::make_unique<ThreeDPresentation>();
            break;       
        default:
            break;
        }
        update();
    });

}

void LinearDimension::update()
{
    Dimension::update();
    if(cover->getScale() != m_oldScale || pinMoving() || m_scalingChanged|| pinPlacing())
    {
        auto p1 = getPinPos(0).getTrans();
        auto p2 = getPinPos(1).getTrans();
        m_presentation->update(p1, p2, m_scale);
        m_oldScale = cover->getScale();
        m_distanceEdit->setValue(p2 - p1);
        m_scalingChanged = false;

    }
}


void LinearDimension::scalingChanged()
{
    m_scalingChanged = true;
}

