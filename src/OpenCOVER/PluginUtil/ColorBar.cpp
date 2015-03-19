/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include "ColorBar.h"

#include <util/common.h>
#include <config/CoviseConfig.h>

using namespace vrui;
using namespace opencover;

static const char *MINMAX = "MinMax";
static const char *STEPS = "numSteps";
static const char *AUTOSCALE = "autoScales";

ColorBar::ColorBar(const char *name, char *species,
                   float min, float max, int numColors,
                   float *r, float *g, float *b, float *a)
{
    name_ = strcpy(new char[strlen(name) + 1], name);
    species_ = strcpy(new char[strlen(species) + 1], species);
    title_ = new char[strlen(name) + strlen(species) + 10];
    if (strcmp(species_, "Colors") == 0)
        strcpy(title_, name);
    else
    {
        //sprintf(title_,"%s:%s", name, species_);
        strcpy(title_, species_);
    }
    colorbar_ = new coColorBar(name, species_, min, max, numColors, r, g, b, a);
    pinboard_ = opencover::cover->getMenu();
    colorsButton_ = new coSubMenuItem(name);
    myColorsButton_ = colorsButton_;
    colorsMenu_ = new coRowMenu(title_, pinboard_);
    pinboard_->add(colorsButton_);
    colorsMenu_->add(colorbar_);
    colorsButton_->setMenu(colorsMenu_);

    tabUI = false;
    minSlider_ = NULL;
    maxSlider_ = NULL;
    stepSlider_ = NULL;
    autoScale_ = NULL;
    execute_ = NULL;
    inter_ = NULL;
}

ColorBar::ColorBar(coSubMenuItem *colorsButton, coRowMenu *moduleMenu, const char *name,
                   char *species, float min, float max, int numColors,
                   float *r, float *g, float *b, float *a, int tabID)
    : colorsButton_(colorsButton)
{
    myColorsButton_ = NULL;
    ;
    name_ = strcpy(new char[strlen(name) + 1], name);
    species_ = strcpy(new char[strlen(species) + 1], species);
    title_ = new char[strlen(name) + strlen(species_) + 10];
    if (strcmp(species, "Colors") == 0)
        strcpy(title_, name);
    else
        strcpy(title_, species_);
    colorbar_ = new coColorBar(name, species_, min, max, numColors, r, g, b, a);
    pinboard_ = NULL;
    colorsMenu_ = new coRowMenu(title_, moduleMenu);
    colorsMenu_->add(colorbar_);

    //coSliderMenuItem * slider;
    float diff = (max - min) / 2;
    float minMin = min - diff;
    float maxMin = min + diff;
    float minMax = max - diff;
    float maxMax = max + diff;
    minSlider_ = new coSliderMenuItem("MIN", minMin, maxMin, min);
    minSlider_->setPrecision(6);
    maxSlider_ = new coSliderMenuItem("MAX", minMax, maxMax, max);
    maxSlider_->setPrecision(6);
    stepSlider_ = new coSliderMenuItem("Steps", 2, numColors, numColors);
    stepSlider_->setInteger(true);

    autoScale_ = new coCheckboxMenuItem("autoScale", false);
    execute_ = new coButtonMenuItem("execute");

    autoScale_->setMenuListener(this);
    execute_->setMenuListener(this);
    minSlider_->setMenuListener(this);
    maxSlider_->setMenuListener(this);
    stepSlider_->setMenuListener(this);

    colorsMenu_->add(minSlider_);
    colorsMenu_->add(maxSlider_);
    colorsMenu_->add(stepSlider_);
    colorsMenu_->add(autoScale_);
    colorsMenu_->add(execute_);

    colorsButton_->setMenu(colorsMenu_);
    inter_ = NULL;

    tabUI = false;
    // create the TabletUI User-Interface
    if (tabID != -1)
    {
        createMenuEntry(name, min, max, numColors, tabID);
        tabUI = true;
    }
}

ColorBar::~ColorBar()
{
    colorsMenu_->remove(colorbar_);

    if (tabUI)
        removeMenuEntry();

    if (minSlider_)
        delete minSlider_;
    if (maxSlider_)
        delete maxSlider_;
    if (stepSlider_)
        delete stepSlider_;
    if (autoScale_)
        delete autoScale_;
    if (execute_)
        delete execute_;

    delete colorbar_;
    delete[] title_;
    delete colorsMenu_;
    colorsButton_->setMenu(NULL);

    if (myColorsButton_) // if constructor without submenuButton was used
    { // the submenuButton was created in this class and can be
        // deleted in this class
        fprintf(stderr, "deleting colorsbutton\n");
        if (pinboard_)
        {
            delete colorsButton_;
            colorsButton_ = myColorsButton_ = NULL;
        }
    }
    delete[] name_;

    if (inter_)
    {
        inter_->decRefCount();
        inter_ = NULL;
    }
}

void ColorBar::createMenuEntry(const char *name, float min, float max, int numColors, int tabID)
{

    // tab
    _tab = new coTUITab(name, tabID);
    _tab->setPos(0, 0);

    _minL = new coTUILabel("Min", _tab->getID());
    _maxL = new coTUILabel("Max", _tab->getID());
    _stepL = new coTUILabel("numSteps", _tab->getID());

    _min = new coTUIEditFloatField("min", _tab->getID());
    _max = new coTUIEditFloatField("max", _tab->getID());
    _steps = new coTUIEditIntField("steps", _tab->getID());

    _minL->setPos(0, 0);
    _min->setPos(1, 0);

    _maxL->setPos(0, 2);
    _max->setPos(1, 2);

    _stepL->setPos(0, 4);
    _steps->setPos(1, 4);

    _min->setValue(min);
    _max->setValue(max);
    _steps->setValue(numColors);

    _autoScale = new coTUIToggleButton("autoScale", _tab->getID());
    _autoScale->setPos(0, 6);

    _execute = new coTUIButton("Execute", _tab->getID());
    _execute->setPos(2, 10);
    _execute->setEventListener(this);
}

void ColorBar::removeMenuEntry()
{

    delete _min;
    delete _max;
    delete _steps;
    delete _minL;
    delete _maxL;
    delete _stepL;

    delete _autoScale;
    delete _execute;
    delete _tab;
}

void ColorBar::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == _execute)
    {
        if (inter_)
        {
            float minmax[2];
            minmax[0] = _min->getValue();
            minmax[1] = _max->getValue();
            inter_->setVectorParam(MINMAX, 2, minmax);

            int num = _steps->getValue();
            inter_->setScalarParam(STEPS, num);

            if (_autoScale->getState())
                inter_->setBooleanParam(AUTOSCALE, 1);
            else
                inter_->setBooleanParam(AUTOSCALE, 0);

            inter_->executeModule();
        }
    }
}

void
ColorBar::update(const char *species, float min, float max, int numColors, float *r, float *g, float *b, float *a)
{
    colorbar_->update(min, max, numColors, r, g, b, a);

    delete[] species_;
    species_ = strcpy(new char[strlen(species) + 1], species);

    delete[] title_;
    title_ = new char[strlen(name_) + strlen(species_) + 10];
    if (strcmp(species_, "Colors") == 0)
        strcpy(title_, name_);
    else
        //sprintf(title_,"%s:%s", name_, species);
        strcpy(title_, species);

    colorsMenu_->updateTitle(title_);

    float diff = (max - min) / 2;
    float minMin = min - diff;
    float maxMin = min + diff;
    float minMax = max - diff;
    float maxMax = max + diff;

    if (minSlider_)
    {
        minSlider_->setMin(minMin);
        minSlider_->setMax(maxMin);
        minSlider_->setValue(min);
        maxSlider_->setMin(minMax);
        maxSlider_->setMax(maxMax);
        maxSlider_->setValue(max);

        if (numColors > stepSlider_->getMax())
        {
            stepSlider_->setMax(numColors);
        }
        stepSlider_->setValue(numColors);
    }

    int state = 0;
    if (inter_)
    {

        inter_->getBooleanParam(AUTOSCALE, state);
        if (state)
            autoScale_->setState(true);
        else
            autoScale_->setState(false);
    }
    if (tabUI)
    {
        _min->setValue(min);
        _max->setValue(max);
        _steps->setValue(numColors);
        if (state)
            _autoScale->setState(true);
        else
            _autoScale->setState(false);
    }
}

void
ColorBar::setName(const char *name)
{
    delete[] name_;
    name_ = new char[strlen(name) + 1];
    strcpy(name_, name);

    delete[] title_;
    title_ = new char[strlen(name_) + strlen(species_) + 10];
    if (strcmp(species_, "Colors") == 0)
        strcpy(title_, name_);
    else
        //sprintf(title_,"%s:%s", name_, species);
        strcpy(title_, species_);

    colorsMenu_->updateTitle(title_);
    colorsButton_->setLabel(name_);
}

const char *
ColorBar::getName()
{
    return colorbar_->getName();
}

void
ColorBar::menuEvent(coMenuItem *menuItem)
{
    (void)menuItem;
    if (inter_)
    {
        if (menuItem == execute_)
        {
            inter_->executeModule();
        }
        if (menuItem == autoScale_)
        {

            if (autoScale_->getState())
                inter_->setBooleanParam(AUTOSCALE, 1);
            else
                inter_->setBooleanParam(AUTOSCALE, 0);
        }
    }
}

void
ColorBar::menuReleaseEvent(coMenuItem *menuItem)
{
    if (inter_)
    {
        if (menuItem == minSlider_ || menuItem == maxSlider_)
        {
            float minmax[2];
            minmax[0] = minSlider_->getValue();
            minmax[1] = maxSlider_->getValue();
            inter_->setVectorParam(MINMAX, 2, minmax);
            //inter_->executeModule();
        }
        if (menuItem == stepSlider_)
        {
            int num = static_cast<int>(stepSlider_->getValue());
            inter_->setScalarParam(STEPS, num);
            //inter_->executeModule();
        }
    }
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
    colorsMenu_->setScale(scale);
    colorsMenu_->setVisible(visible);
}

bool ColorBar::isVisible()
{
    return colorsMenu_->isVisible();
}
