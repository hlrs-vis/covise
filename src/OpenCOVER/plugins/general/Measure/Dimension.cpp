
#include "Dimension.h"
#include "Measure.h"

#include <cover/coVRFileManager.h>
#include <cover/coBillboard.h>

#include <cover/ui/Slider.h>

#include <config/CoviseConfig.h>

using namespace opencover;

Scaling::Scaling(coVRPlugin *plugin, ui::Group *parent)
: m_plugin(plugin)
, m_coneSize(new ui::Slider(parent, "coneSize"))
, m_fontFactor(new ui::Slider(parent, "fontFactor"))
, m_lineWidth(new ui::Slider(parent, "lineWidth"))
{
    m_coneSize->setValue(plugin->configFloat("Cone", "size", 150.0)->value());
    m_fontFactor->setValue(plugin->configFloat("Text", "size", 3.0)->value());
    m_lineWidth->setValue(plugin->configFloat("Text", "lineWidth", 28.0)->value());

    m_coneSize->setPresentation(ui::Slider::Presentation::AsDial);
    m_fontFactor->setPresentation(ui::Slider::Presentation::AsDial);
    m_lineWidth->setPresentation(ui::Slider::Presentation::AsDial);

    m_coneSize->setCallback([this](ui::Slider::ValueType, bool){
        if(m_callback)
            m_callback();
    });
    m_fontFactor->setCallback([this](ui::Slider::ValueType, bool){
        if(m_callback)
            m_callback();
    });
    m_lineWidth->setCallback([this](ui::Slider::ValueType, bool){
        if(m_callback)
            m_callback();
    });

    m_coneSize->setBounds(0, 600);
    m_fontFactor->setBounds(0, 10);
    m_lineWidth->setBounds(1, 30);
}

Scaling &Scaling::operator=(const Scaling &other)
{
    m_coneSize->setValue(other.m_coneSize->value());
    m_fontFactor->setValue(other.m_fontFactor->value());
    m_lineWidth->setValue(other.m_lineWidth->value());
    return *this;
}

Scaling::~Scaling()
{
    auto parent = m_coneSize->parent();
    parent->remove(m_coneSize);
    parent->remove(m_fontFactor);
    parent->remove(m_lineWidth);
}

float Scaling::coneSize() const
{
    return m_coneSize->value();
}

float Scaling::lineWidth() const
{
    return m_lineWidth->value();
}

float Scaling::fontFactor() const
{
    return m_fontFactor->value();
}

void Scaling::setCallback(const std::function<void()> &onScalingChanged)
{
    m_callback = onScalingChanged;
}

TextBoard::TextBoard()
{
    m_text = new osgText::Text();
    m_text->setName("MeasureText");
    m_text->setDataVariance(osg::Object::DYNAMIC);
    m_text->setFont(coVRFileManager::instance()->getFontFile(NULL));
    m_text->setDrawMode(osgText::Text::TEXT);

    m_text->setColor(osg::Vec4(1, 1, 1, 1));

    m_text->setAlignment(osgText::Text::CENTER_BASE_LINE);

    m_text->setCharacterSize(40.0);
    m_text->setLayout(osgText::Text::LEFT_TO_RIGHT);
    m_text->setAxisAlignment(osgText::Text::XY_PLANE);
    setText("0.0");

    osg::ref_ptr<osg::Geode> textNode = new osg::Geode();
    textNode->addDrawable(m_text);

    coBillboard *billBoard = new coBillboard();
    billBoard->setMode(coBillboard::AXIAL_ROT);
    billBoard->setMode(coBillboard::POINT_ROT_WORLD);

    osg::Vec3 zaxis(0, 1, 0);
    //  osg::Vec3 zaxis = -cover->getViewerMat().getTrans();
    billBoard->setAxis(zaxis);

    osg::Vec3 normal(0, 0, 1);

    billBoard->setNormal(normal);
    billBoard->addChild(textNode.get());
    m_transform = new osg::MatrixTransform();
    m_transform->addChild(billBoard);
    cover->getObjectsRoot()->addChild(m_transform);
    m_transform->setMatrix(osg::Matrix::scale(osg::Vec3f{0.001,0.001,0.001}));
}

TextBoard::~TextBoard()
{
    cover->getObjectsRoot()->removeChild(m_transform);
}

void TextBoard::setText(const std::string &text)
{
    m_text->setText(text, osgText::String::ENCODING_UTF8);
}

void TextBoard::setPosition(const osg::Matrix &pos)
{
    m_transform->setMatrix(pos);
}

void TextBoard::setPosition(const osg::Vec3f &pos)
{
    auto m = m_transform->getMatrix();
    m.setTrans(pos); 
    m_transform->setMatrix(m);
}


void Dimension::setScaling(const Scaling &scaling)
{
    m_scale = scaling;
    for(auto &pin : m_pins)
        pin->setConeSize(m_scale.coneSize());
    scalingChanged();
}

std::chrono::system_clock::time_point Dimension::getLastChange() const
{
    std::chrono::system_clock::time_point t;
    for(const auto &pin : m_pins)
    {
        if(pin->getTimeOfLastChange() > t)
            t = pin->getTimeOfLastChange();
    }
    return t;
}

void Dimension::scalingChanged()
{
 // use overload to do sth
}

osg::Matrix Dimension::getPinPos(int index)
{
    return m_pins[index]->getMat();
}

void Dimension::addPin()
{
    m_pins.push_back(std::make_unique<Pin>(m_scale.coneSize(), m_pins.size(), m_id, m_gui.get()));
}

bool Dimension::pinPlacing()
{
    bool placing = false;
    for(const auto &pin : m_pins)
        placing |= pin->placing;
    return placing;
}

bool Dimension::pinMoving()
{
    bool moving = false;
    for(const auto &pin : m_pins)
    {
        moving |= pin->moveMarker;
        moving |= pin->getTimeOfLastChange() > m_lastPinCheck;
    }
    m_lastPinCheck = std::chrono::system_clock::now();
    return moving;
}

bool Dimension::isplaced()
{
    for (auto &mark : m_pins)
    {
        if (mark->placing)
            return true;
    }
    return false;
}

Dimension::Dimension(int id, const std::string &name, coVRPlugin *plugin, ui::Group *parent, const Scaling &scale)
: m_plugin(plugin)
, m_gui(new ui::Menu(parent, name))
, m_id(id)
, m_scale(plugin, m_gui.get())
{
    m_scale = scale;
    m_scale.setCallback([this](){
        update();
        for (auto &pin : m_pins)
            pin->setConeSize(m_scale.coneSize());
        scalingChanged();
    });
}

void Dimension::update()
{
    for (auto &pin : m_pins)
        pin->update();
}