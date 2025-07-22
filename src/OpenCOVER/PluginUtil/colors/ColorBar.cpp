/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include "ColorBar.h"

#include <util/common.h>
#include <config/CoviseConfig.h>

#include <cover/ui/Menu.h>
#include <cover/ui/VruiView.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SpecialElement.h>
#include <cover/ui/SelectionList.h>

#include <cover/VRVruiRenderInterface.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/mathUtils.h>

static const char MINMAX[] = "MinMax";
static const char STEPS[] = "numSteps";
static const char AUTOSCALE[] = "autoScales";

static const char V_STEPS[] = "steps";
static const char V_MIN[] = "min";
static const char V_MAX[] = "max";
static const char V_AUTOSCALE[] = "auto_range";
static const char V_CENTER[] = "center";
static const char V_COMPRESS[] = "range_compression";

static const char V_NEST[] = "nest";
static const char V_INSETAUTOCENTER[] = "auto_center";
static const char V_INSETRELATIVE[] = "inset_relative";
static const char V_INSETCENTER[] = "inset_center";
static const char V_INSETWIDTH[] = "inset_width";
static const char V_OPACITYFACTOR[] = "opacity_factor";
static const char V_BLENDWITHMATERIAL[] = "blend_with_material";

using namespace  vrui;
using namespace  opencover;

// namespace opencover
// {

    // opencover::ColorMap interpolateColorMap(const opencover::ColorMap &cm)
    // {
    //     if(cm.steps()  == cm.r.size())
    //         return cm;
    //     int numSteps = cm.steps;
    //     opencover::ColorMap interpolatedMap = cm;
    //     interpolatedMap.r.resize(numSteps);
    //     interpolatedMap.g.resize(numSteps);
    //     interpolatedMap.b.resize(numSteps);
    //     interpolatedMap.a.resize(numSteps);
    //     interpolatedMap.samplingPoints.resize(numSteps);

    //     auto numColors = cm.samplingPoints.size();
    //     double delta = 1.0 / (numSteps - 1);
    //     int idx = 0;

    //     for (int i = 0; i < numSteps - 1; i++)
    //     {
    //         double x = i * delta;
    //         while (cm.samplingPoints[idx + 1] <= x)
    //         {
    //             idx++;
    //             if (idx > numColors - 2)
    //             {
    //                 idx = numColors - 2;
    //                 break;
    //             }
    //         }

    //         double d = (x - cm.samplingPoints[idx]) / (cm.samplingPoints[idx + 1] - cm.samplingPoints[idx]);
    //         interpolatedMap.r[i] = static_cast<float>((1 - d) * cm.r[idx] + d * cm.r[idx + 1]);
    //         interpolatedMap.g[i] = static_cast<float>((1 - d) * cm.g[idx] + d * cm.g[idx + 1]);
    //         interpolatedMap.b[i] = static_cast<float>((1 - d) * cm.b[idx] + d * cm.b[idx + 1]);
    //         interpolatedMap.a[i] = static_cast<float>((1 - d) * cm.a[idx] + d * cm.a[idx + 1]);
    //         interpolatedMap.samplingPoints[i] = static_cast<float>(i) / (numSteps - 1);
    //     }

    //     interpolatedMap.r[numSteps - 1] = cm.r[numColors - 1];
    //     interpolatedMap.g[numSteps - 1] = cm.g[numColors - 1];
    //     interpolatedMap.b[numSteps - 1] = cm.b[numColors - 1];
    //     interpolatedMap.a[numSteps - 1] = cm.a[numColors - 1];
    //     interpolatedMap.samplingPoints[numSteps - 1] = 1.0f;

    //     interpolatedMap.steps = numSteps;

    //     return interpolatedMap;
    // }

// ColorBar::ColorBar(ui::Menu *menu)
// : ui::Owner(std::string("ColorBar"), menu)
// , colorsMenu_(menu)
// , map_(menu->name())
// {
//     init();
// }

ColorBar::ColorBar(ui::Group *group)
: ui::Owner(std::string("ColorBar"), group)
, colorsMenu_(group)
, map_(group->name())
{
    init();
}

void ColorBar::init()
{
    show_ = new ui::Button("Show", this);
    colorsMenu_->add(show_);
    show_->setVisible(false, ui::View::VR);
    show_->setState(false);
    show_->setCallback([this](bool state){
        show(state);
    });

    if (cover->vruiView)
    {
        uiColorBar_ = new ui::SpecialElement("VruiColorBar", this);

        colorsMenu_->add(uiColorBar_);
        uiColorBar_->registerCreateDestroy(cover->vruiView->typeBit(),
                [this](ui::SpecialElement *se, ui::View::ViewElement *ve){
                auto vve = dynamic_cast<ui::VruiViewElement *>(ve);
                assert(vve);
                colorbar_ = std::make_unique<coColorBar>(name_, map_);
                vve->m_menuItem = colorbar_.get();
                },
                [this](ui::SpecialElement *se, ui::View::ViewElement *ve){
                auto vve = dynamic_cast<ui::VruiViewElement *>(ve);
                assert(vve);
                assert(!colorbar_ || !vve->m_menuItem || vve->m_menuItem == colorbar_.get());
                colorbar_.reset();
                vve->m_menuItem = nullptr;
                });
    }

    autoScale_ = new ui::Button("AutoRange", this);
    colorsMenu_->add(autoScale_);
    autoScale_->setText("Auto range");
    autoScale_->setState(false);


    float diff = (map_.max() - map_.min()) / 2;
    minSlider_ = new ui::Slider("Min", this);
    colorsMenu_->add(minSlider_);
    minSlider_->setBounds(map_.min()-diff, map_.min()+diff);
    minSlider_->setValue(map_.min());


    maxSlider_ = new ui::Slider("Max", this);
    colorsMenu_->add(maxSlider_);
    maxSlider_->setBounds(map_.max()-diff, map_.max()+diff);
    maxSlider_->setValue(map_.max());


    center_ = new ui::Slider("Center", this);
    colorsMenu_->add(center_);
    center_->setBounds(0., 1.);
    center_->setValue(0.5);


    compress_ = new ui::Slider("RangeCompression", this);
    colorsMenu_->add(compress_);
    compress_->setText("Range compression");
    compress_->setBounds(-1, 1);
    compress_->setValue(0);


    stepSlider_ = new ui::Slider("Steps", this);
    colorsMenu_->add(stepSlider_);
    stepSlider_->setBounds(2, map_.steps());
    stepSlider_->setIntegral(true);
    stepSlider_->setScale(ui::Slider::Logarithmic);


    insetCenter_ = new ui::Slider("InsetCenter", this);
    colorsMenu_->add(insetCenter_);
    insetCenter_->setText("Inset center");
    insetCenter_->setBounds(0, 1);
    insetCenter_->setValue(0.5);


    insetWidth_ = new ui::Slider("InsetWidth", this);
    colorsMenu_->add(insetWidth_);
    insetWidth_->setText("Inset width");
    insetWidth_->setBounds(0., 1.);
    insetWidth_->setValue(0.1);


    opacityFactor_ = new ui::Slider("OpacityFactor", this);
    colorsMenu_->add(opacityFactor_);
    opacityFactor_->setText("Opacity factor");
    opacityFactor_->setBounds(0., 1.);
    opacityFactor_->setValue(1.);

    updateTitle();
}

bool ColorBar::hudVisible() const
{
    return hudbar_ && hudbar_->isVisible();
}

ColorBar::HudPosition::HudPosition(float hudScale)
{
    if (coVRMSController::instance()->isMaster() && coVRConfig::instance()->numScreens() > 0) {
        const auto &s0 = coVRConfig::instance()->screens[0];
        hpr = s0.hpr;
        auto sz = osg::Vec3(s0.hsize, 0., s0.vsize);
        osg::Matrix mat;
        MAKE_EULER_MAT_VEC(mat, hpr);
        m_bottomLeft = s0.xyz - sz * mat * 0.5;
        auto minsize = std::min(s0.hsize, s0.vsize);
        m_bottomLeft += osg::Vec3(minsize, 0., minsize) * mat * 0.02;
        m_offset = osg::Vec3(s0.vsize/2.5, 0 , 0) * mat * hudScale;
    }
    for (int i=0; i<3; ++i)
    {
        coVRMSController::instance()->syncData(&m_bottomLeft[i], sizeof(m_bottomLeft[i]));
        coVRMSController::instance()->syncData(&hpr[i], sizeof(hpr[i]));
        coVRMSController::instance()->syncData(&m_offset[i], sizeof(m_offset[i]));
    }
    scale = m_offset[0]/480;
    bottomLeft = m_bottomLeft;
}

void ColorBar::HudPosition::setNumHuds(int numHuds)
{
    bottomLeft = m_bottomLeft + m_offset * numHuds;
}

void ColorBar::setHudPosition(const HudPosition &pos)
{
    if (!hudbar_)
        return;

    auto mat = coUIElement::getMatrixFromPositionHprScale(pos.bottomLeft[0], pos.bottomLeft[1], pos.bottomLeft[2], pos.hpr[0], pos.hpr[1], pos.hpr[2], pos.scale);
    auto uie = hudbar_->getUIElement();
    auto vtr = uie->getDCS();
    vtr->setMatrix(mat);
    vruiRendererInterface::the()->deleteMatrix(mat);
}

void ColorBar::updateTitle()
{
    title_ = name_;
    if (map_.species() != "Color" && !map_.species().empty())
    {
        title_ += ": " + map_.species();
    }
    if(!map_.unit().empty() && map_.unit() != "NoUnit")
    {
        title_ += " (" + map_.unit() + ")";
    }
    colorsMenu_->setText(title_);
}

void ColorBar::displayColorMap()
{
    if (colorbar_)
        colorbar_->update(map_);
    if (hudbar_)
        hudbar_->update(map_);
    stepSlider_->setValue(map_.steps());
    minSlider_->setValue(map_.min());
    maxSlider_->setValue(map_.max());
}

// void
// ColorBar::update(const ColorMap &map)
// {
//     map_ = map;
//     updateTitle();

//     displayColorMap();
//     update();
    
// }

void CoviseColorBar::updateGui()
{
    if (stepSlider_)
    {
        int imin=0, imax=0, ival=0;
        if (!inter_ || inter_->getIntSliderParam(V_STEPS, imin, imax, ival) == -1)
        {
            if (map_.steps() > stepSlider_->max())
            {
                stepSlider_->setBounds(2, map_.steps());
            }
            stepSlider_->setValue(map_.steps());
        }
    }

    if (minSlider_ && maxSlider_)
    {
        float smin = 0.f, smax = 0.f, sval = 0.f;
        if (!inter_
            || inter_->getFloatSliderParam(V_MIN, smin, smax, sval) == -1
            || inter_->getFloatSliderParam(V_MAX, smin, smax, sval) == -1)
        {
            float diff = (map_.max() - map_.min()) / 2;

            minSlider_->setBounds(map_.min()-diff, map_.min()+diff);
            maxSlider_->setBounds(map_.max()-diff, map_.max()+diff);
            if(!inter_)
            {
                minSlider_->setValue(map_.min());
                maxSlider_->setValue(map_.max());
            }
        }
    }
}

void CoviseColorBar::updateInteractor()
{
    if (!inter_)
        return;

    if (stepSlider_)
    {
        int imin=0, imax=0, ival=0;
        if (inter_->getIntSliderParam(V_STEPS, imin, imax, ival) != -1)
        {
            stepSlider_->setBounds(imin, imax);
            stepSlider_->setValue(ival);
        }
    }

    int num = 0;
    float *minmax = nullptr;
    if (inter_->getFloatVectorParam(MINMAX, num, minmax) != -1 && num == 2)
    {
        if (minSlider_)
        {
            minSlider_->setValue(minmax[0]);
        }

        if (maxSlider_)
        {
            maxSlider_->setValue(minmax[1]);
        }
    }

    float smin = 0.f, smax = 0.f, sval = 0.f;
    if (minSlider_ && inter_->getFloatSliderParam(V_MIN, smin, smax, sval) != -1)
    {
        minSlider_->setBounds(smin, smax);
        minSlider_->setValue(sval);

    }

    if (maxSlider_ && inter_->getFloatSliderParam(V_MAX, smin, smax, sval) != -1)
    {
        maxSlider_->setBounds(smin, smax);
        maxSlider_->setValue(sval);
    }

    int state = 0;
    if (inter_->getBooleanParam(AUTOSCALE, state) == -1)
        inter_->getBooleanParam(V_AUTOSCALE, state);
    if (state)
        autoScale_->setState(true);
    else
        autoScale_->setState(false);

    int nest = 0;
    if (inter_->getBooleanParam(V_NEST, nest) == -1)
        nest = 0;

    float center = 0.;
    if (center_ && inter_->getFloatScalarParam(V_CENTER, center) != -1)
    {
        center_->setValue(center);
        center_->setVisible(nest == 0);
    }
    else
    {
        center_->setVisible(false);
    }

    float compress = 0.;
    if (compress_ && inter_->getFloatScalarParam(V_COMPRESS, compress) != -1)
    {
        compress_->setValue(compress);
        compress_->setVisible(nest == 0);
    }
    else
    {
        compress_->setVisible(false);
    }

    int autocenter = 1;
    if (inter_->getBooleanParam(V_INSETAUTOCENTER, autocenter) == -1)
        autocenter = 1;
    int inset_rel = 0;
    if (inter_->getBooleanParam(V_INSETRELATIVE, inset_rel) == -1)
        inset_rel = 0;

    float insetCenter = 0.;
    if (insetCenter_ && inter_->getFloatScalarParam(V_INSETCENTER, insetCenter) != -1)
    {
        insetCenter_->setValue(insetCenter);
        insetCenter_->setVisible(inset_rel == 1 && nest != 0 && autocenter == 0);
    }
    else
    {
        insetCenter_->setVisible(false);
    }

    float insetWidth = 0.;
    if (insetWidth_ && inter_->getFloatScalarParam(V_INSETWIDTH, insetWidth) != -1)
    {
        insetWidth_->setValue(insetWidth);
        insetWidth_->setVisible(inset_rel == 1 && nest != 0);
    }
    else
    {
        insetWidth_->setVisible(false);
    }

    int blendWithMaterial = 0;
    if (inter_->getBooleanParam(V_BLENDWITHMATERIAL, blendWithMaterial) == -1)
        blendWithMaterial = 0;

    float opacityFactor = 1.;
    if (opacityFactor_ && inter_->getFloatScalarParam(V_OPACITYFACTOR, opacityFactor) != -1)
    {
        opacityFactor_->setValue(opacityFactor);
        opacityFactor_->setVisible(blendWithMaterial != 0);
    }
    else
    {
        opacityFactor_->setVisible(false);
    }
}

void CoverColorBar::setCallback(const std::function<void(const ColorMap &)> &f)
{
  m_callback = f;
}

void ColorBar::setMinBounds(float min, float max)
{
    minSlider_->setBounds(min, max);
}

void ColorBar::setMaxBounds(float min, float max)
{
    maxSlider_->setBounds(min, max);
}

void ColorBar::setMaxNumSteps(int maxSteps)
{
    stepSlider_->setBounds(2, maxSteps);
}

void
ColorBar::setName(const std::string &name)
{
    name_ = name;
    updateTitle();
}

void ColorBar::show(bool state)
{
    if (state)
    {
        if(!hudbar_)
        {
            hudbar_ = std::make_unique<coColorBar>(name_, map_);
            hudbar_->getUIElement()->createGeometry();
        }
        auto vtr = hudbar_->getUIElement()->getDCS();
        VRVruiRenderInterface::the()->getAlwaysVisibleGroup()->addChild(vtr);
    }
    if(hudbar_)
        hudbar_->setVisible(state);
    show_->setState(state);
}

const char *
ColorBar::getName()
{
    if (colorbar_)
        return colorbar_->getName();

    return "";
}

void
CoviseColorBar::addInter(coInteractor *inter)
{
    inter->incRefCount();
    if (inter_)
    {
        inter_->decRefCount();
        inter_ = NULL;
    }
    inter_ = inter;

    updateInteractor();
}

void ColorBar::setVisible(bool visible)
{
    float scale;
    std::string sizeStr = covise::coCoviseConfig::getEntry("AKToolbar.Scale");
    if (!sizeStr.empty())
        sscanf(sizeStr.c_str(), "%5f", &scale);
    else
        scale = 0.2;
    //colorsMenu_->setScale(scale);
    colorsMenu_->setVisible(visible);
}

bool ColorBar::isVisible()
{
    return colorsMenu_->visible();
}



CoviseColorBar::CoviseColorBar(ui::Group *menu)
: ColorBar(menu)
{

    autoScale_->setCallback([this](bool state){
        inter_->setBooleanParam(AUTOSCALE, state);
        inter_->setBooleanParam(V_AUTOSCALE, state);
    });

    minSlider_->setCallback([this](double value, bool released){
        map_.setMinMax(value, map_.max());
        if (!inter_)
            return;
        float dummy = 0.;
        if (inter_->getFloatScalarParam(V_MIN, dummy) != -1)
            inter_->setScalarParam(V_MIN, static_cast<float>(value));
        float minmax[2];
        minmax[0] = value;
        minmax[1] = maxSlider_->value();
        inter_->setVectorParam(MINMAX, 2, minmax);
    });

    maxSlider_->setCallback([this](double value, bool released){
        map_.setMinMax(map_.min(), value);
        if (!inter_)
            return;
        float dummy = 0.;
        if (inter_->getFloatScalarParam(V_MAX, dummy) != -1)
            inter_->setScalarParam(V_MAX, static_cast<float>(value));
        float minmax[2];
        minmax[0] = minSlider_->value();
        minmax[1] = value;
        inter_->setVectorParam(MINMAX, 2, minmax);
    });

    center_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_CENTER, static_cast<float>(value));
    });

    compress_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_COMPRESS, static_cast<float>(value));
    });

    stepSlider_->setCallback([this](double value, bool released){
        auto steps = static_cast<int>(value);
        map_.setSteps(steps);
        if (!inter_)
            return;
        inter_->setScalarParam(STEPS, steps);
        inter_->setScalarParam(V_STEPS, steps);
        //inter_->executeModule();
    });
    insetCenter_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_INSETCENTER, static_cast<float>(value));
    });

    insetWidth_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_INSETWIDTH, static_cast<float>(value));
    });

    opacityFactor_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_OPACITYFACTOR, static_cast<float>(value));
    });

    execute_ = new ui::Action("Execute", this);
    colorsMenu_->add(execute_);
    execute_->setCallback([this](){
        if (inter_)
            inter_->executeModule();
    });
}

CoviseColorBar::~CoviseColorBar()
{
    if (inter_)
    {
        inter_->decRefCount();
        inter_ = NULL;
    }
}

void CoviseColorBar::updateFromAttribute(const char *attrib)
{
    map_ = parseAttribute(attrib);
    updateGui();
    displayColorMap();
}

ColorMap CoviseColorBar::parseAttribute(const char *attrib)
{
    int bufLen = strlen(attrib) + 1;
    istringstream attribs(attrib);
    
    // COLORS_1_OUT_001 pressure min max ncolors 0 r g b rgb rgb ....
    std::vector<char> s(bufLen);
    attribs.getline(s.data(), bufLen, '\n'); // overread obj name
    attribs.getline(s.data(), bufLen, '\n'); // read species
    
    int v = 0;
    int numColors = 0;
    float min, max;
    attribs >> min >> max >> numColors >> v;
    std::vector<osg::Vec4> colors(numColors);
    std::vector<float> samplingPoints(numColors);

    
    for (int i = 0; i < numColors; i++)
    {
        attribs >> colors[i].r() >> colors[i].g() >> colors[i].b() >> colors[i].a();
        samplingPoints[i] = static_cast<float>(i) / (numColors - 1);
    }
    return ColorMap(BaseColorMap{colors, samplingPoints, s.data()}, min, max); 
}


CoverColorBar::CoverColorBar(ui::Group *menu)
: ColorBar(menu)
{

    minSlider_->setCallback([this](double value, bool released){
        int min = static_cast<int>(value);
        if(min == map_.min())
            return;
        if(m_callback)
        {
            map_.setMinMax(min, map_.max());
            displayColorMap();
            m_callback(map_);
        }
    });
    maxSlider_->setCallback([this](double value, bool released){
        int max = static_cast<int>(value);
        if(max == map_.max())
            return;
        if(m_callback)
        {
            map_.setMinMax(map_.min(), max);
            displayColorMap();
            m_callback(map_);
        }
    });
    stepSlider_->setCallback([this](double value, bool released){
        auto steps = static_cast<int>(value);
        if(steps == map_.steps())
            return;
        if(m_callback)
        {
            // interpolatedMap_ = opencover::interpolateColorMap(selectedMap_, static_cast<int>(value));
            map_.setSteps(steps);
            displayColorMap();
            m_callback(map_);
        }
    });
    m_selector = new ui::SelectionList("ColorMapSelector", this);
    colorsMenu_->add(m_selector);
    auto &maps = ConfigColorMaps();
    std::vector<std::string> names;
    names.reserve(maps.size());
    for (const auto &map : maps)
        names.push_back(map.name);
    m_selector->setList(names);
    m_selector->setCallback([this](int index){
        if (index < 0 || index >= static_cast<int>(m_selector->items().size()))
            return;
        auto &maps = ConfigColorMaps();
        if (index < static_cast<int>(maps.size()))
        {
            auto species = map_.species();
            auto unit = map_.unit();
            auto steps = map_.steps();
            map_ = ColorMap(maps[index], map_.min(), map_.max());
            map_.setSpecies(species);
            map_.setUnit(unit);
            map_.setSteps(steps);
            displayColorMap();
            if(m_callback)
                m_callback(map_);
        }
    });

}

void CoverColorBar::setMinMax(float min, float max, bool autoBounds)
{
    map_.setMinMax(min, max);
    if(autoBounds)
    {
        auto halfSpan = (max - min) / 2;
        setMinBounds(min - halfSpan, min + halfSpan);
        setMaxBounds(max - halfSpan, max + halfSpan);
    }
    displayColorMap();
}
void CoverColorBar::setSteps(int steps)
{
    map_.setSteps(steps);
    displayColorMap();
}
void CoverColorBar::setSpecies(const std::string &species)
{
    map_.setSpecies(species);
    displayColorMap();
    updateTitle();
}
void CoverColorBar::setUnit(const std::string &unit)
{
    map_.setUnit(unit);
    displayColorMap();
    updateTitle();
}

const ColorMap &CoverColorBar::colorMap() const { return map_; }

void CoverColorBar::setColorMap(const std::string& name)
{
    auto &maps = ConfigColorMaps();
    auto it = std::find_if(maps.begin(), maps.end(), [&name](const BaseColorMap &map) {
        return map.name == name;
    });
    if (it != maps.end())
    {
        map_ = ColorMap(*it, map_.min(), map_.max());
        displayColorMap();
    }
}

