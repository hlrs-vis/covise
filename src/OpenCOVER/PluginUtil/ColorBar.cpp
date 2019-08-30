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

static const char MINMAX[] = "MinMax";
static const char STEPS[] = "numSteps";
static const char AUTOSCALE[] = "autoScales";

static const char V_STEPS[] = "steps";
static const char V_MIN[] = "min";
static const char V_MAX[] = "max";
static const char V_AUTOSCALE[] = "auto_range";

using namespace  vrui;

namespace opencover
{

ColorBar::ColorBar(ui::Menu *menu, char *species, float min, float max, int numColors, float *r, float *g, float *b, float *a)
: ui::Owner(std::string("ColorBar")+species, menu)
{
    title_ = species;
    colorsMenu_ = menu;

    float diff = (max - min) / 2;
    float minMin = min - diff;
    float maxMin = min + diff;
    float minMax = max - diff;
    float maxMax = max + diff;
    minSlider_ = new ui::Slider(colorsMenu_, "Min");
    minSlider_->setBounds(minMin, maxMin);
    minSlider_->setValue(min);
    minSlider_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_MIN, static_cast<float>(value));
        float minmax[2];
        minmax[0] = value;
        minmax[1] = maxSlider_->value();
        inter_->setVectorParam(MINMAX, 2, minmax);
    });

    maxSlider_ = new ui::Slider(colorsMenu_, "Max");
    maxSlider_->setBounds(minMax, maxMax);
    maxSlider_->setValue(max);
    maxSlider_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        inter_->setScalarParam(V_MAX, static_cast<float>(value));
        float minmax[2];
        minmax[0] = minSlider_->value();
        minmax[1] = value;
        inter_->setVectorParam(MINMAX, 2, minmax);
    });

    stepSlider_ = new ui::Slider(colorsMenu_, "Steps");
    stepSlider_->setBounds(1, numColors);
    stepSlider_->setIntegral(true);
    stepSlider_->setCallback([this](double value, bool released){
        if (!inter_)
            return;
        int num = static_cast<int>(value);
        inter_->setScalarParam(STEPS, num);
        inter_->setScalarParam(V_STEPS, num);
        //inter_->executeModule();
    });

    autoScale_ = new ui::Button(colorsMenu_, "AutoRange");
    autoScale_->setText("Auto range");
    autoScale_->setState(false);
    autoScale_->setCallback([this](bool state){
        if (!inter_)
            return;
        inter_->setBooleanParam(AUTOSCALE, state);
        inter_->setBooleanParam(V_AUTOSCALE, state);
    });

    execute_ = new ui::Action(colorsMenu_, "Execute");
    execute_->setCallback([this](){
        if (inter_)
            inter_->executeModule();
    });

    if (cover->vruiView)
    {
        auto menu = dynamic_cast<vrui::coRowMenu *>(cover->vruiView->getMenu(colorsMenu_));
        auto item = dynamic_cast<vrui::coSubMenuItem *>(cover->vruiView->getItem(colorsMenu_));
        if (menu && item)
        {
            colorbar_ = new coColorBar(name_.c_str(), species_.c_str(), min, max, numColors, r, g, b, a);
            menu->add(colorbar_);
        }
    }
}

ColorBar::~ColorBar()
{
    delete colorbar_;

    if (inter_)
    {
        inter_->decRefCount();
        inter_ = NULL;
    }
}

void ColorBar::updateTitle()
{
    if (species_ == "Color" || species_.empty())
    {
        title_ = name_;
    }
    else
    {
        title_ = species_;
    }
    colorsMenu_->setText(title_);
}


void
ColorBar::update(const char *species, float min, float max, int numColors, float *r, float *g, float *b, float *a)
{
    if (colorbar_)
        colorbar_->update(min, max, numColors, r, g, b, a);

    if (species)
        species_ = species;
    else
        species_.clear();
    updateTitle();

    float diff = (max - min) / 2;
    float minMin = min - diff;
    float maxMin = min + diff;
    float minMax = max - diff;
    float maxMax = max + diff;

    if (minSlider_)
    {
        minSlider_->setBounds(minMin, maxMin);
        minSlider_->setValue(min);
        maxSlider_->setBounds(minMax, maxMax);
        maxSlider_->setValue(max);

        if (numColors > stepSlider_->max())
        {
            stepSlider_->setBounds(1, numColors);
        }
        stepSlider_->setValue(numColors);
    }

    int state = 0;
    if (inter_)
    {

        if (inter_->getBooleanParam(AUTOSCALE, state) == -1)
            inter_->getBooleanParam(V_AUTOSCALE, state);
        if (state)
            autoScale_->setState(true);
        else
            autoScale_->setState(false);
    }
}

void
ColorBar::setName(const char *name)
{
    if (name)
        name_ = name;
    else
        name_.clear();

    updateTitle();
}

const char *
ColorBar::getName()
{
    return colorbar_->getName();
}

void
ColorBar::addInter(coInteractor *inter)
{
    if (inter_)
    {
        inter_->decRefCount();
        inter_ = NULL;
    }
    inter_ = inter;
    inter_->incRefCount();
}

void
ColorBar::parseAttrib(const char *attrib, char *&species,
                      float &min, float &max, int &numColors,
                      float *&r, float *&g, float *&b, float *&a)
{
    // convert to a istringstream
    int bufLen = strlen(attrib) + 1;
    char *buffer = strcpy(new char[bufLen], attrib);
    istringstream attribs(buffer);

    //fprintf(stderr,"colorsPlugin::addColorbar [%s]\n", name);

    // COLORS_1_OUT_001 pressure min max ncolors 0 r g b rgb rgb ....
    species = new char[bufLen];
    int i;

    attribs.getline(species, bufLen, '\n'); // overread obj name
    attribs.getline(species, bufLen, '\n'); // read species
    attribs >> min >> max >> numColors >> i;

    r = new float[numColors];
    g = new float[numColors];
    b = new float[numColors];
    a = new float[numColors];

    for (i = 0; i < numColors; i++)
    {
        attribs >> r[i] >> g[i] >> b[i] >> a[i];
        //      a[i]=1.0f;
    }

    delete[] buffer;
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

}
