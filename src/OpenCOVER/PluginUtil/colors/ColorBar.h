/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLOR_BAR_H_
#define _COLOR_BAR_H_

#include "coColorBar.h"

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>
#include <cover/coVRTui.h>

#include <util/coTypes.h>
#include <cover/coInteractor.h>

#include <functional>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// class ColorBar
//
// ColorBar manages a coColorbar, the submenu in which the colorbar appears
// and the button which opens/closes the submenu
//
// Initial version: 2002-02, dr
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by Vircinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
namespace opencover
{
namespace ui
{
class SpecialElement;
class SelectionList;
}
// for now colorMaps can only have the configured sample colors, colorMap::getColor interpolates the color for the number of colorMaps's steps   
// this might be inefficient for a lot aof getColorCalls
// ColorMap PLUGIN_UTILEXPORT interpolateColorMap(const ColorMap &cm); //creates a color map with numSteps dedicated colors

class PLUGIN_UTILEXPORT ColorBar: public ui::Owner
{
protected:
    vrui::vruiMatrix *floatingMat_ = nullptr;
    std::unique_ptr<coColorBar> colorbar_, hudbar_;
    ui::Group *colorsMenu_ = nullptr;
    std::string title_;
    std::string name_;
    ui::SpecialElement *uiColorBar_ = nullptr;
    ui::Slider *minSlider_ = nullptr;
    ui::Slider *maxSlider_ = nullptr;
    ui::Slider *stepSlider_ = nullptr;
    ui::Button *autoScale_ = nullptr;
   //  ui::Action *execute_ = nullptr;
    ui::Slider *center_ = nullptr;
    ui::Slider *compress_ = nullptr;
    ui::Slider *insetCenter_ = nullptr;
    ui::Slider *insetWidth_ = nullptr;
    ui::Slider *opacityFactor_ = nullptr;
    ui::Button *show_ = nullptr;
    ColorMap map_;
   void displayColorMap();
   void updateTitle();

private:
    void init();
public:

   //  ColorBar(ui::Menu *menu);
    ColorBar(ui::Group *group);
    virtual ~ColorBar() = default;

    bool hudVisible() const;
    struct PLUGIN_UTILEXPORT HudPosition
    {
      HudPosition(float hudScale = 1.f);
      void setNumHuds(int numHuds); //changes the offset so that all huds are visible
      osg::Vec3 bottomLeft, hpr;
      float scale;
   private:
      osg::Vec3 m_bottomLeft, m_offset;
    };
    void setHudPosition(const HudPosition &pos);

   //  void update(const ColorMap &map);

    void setName(const std::string &name);
    void show(bool state);
    /** get name
     *  @return name the name of the colorbar, identical with module name, eg, g, Colors_1
     */
    const char *getName();

    void setVisible(bool);
    bool isVisible();

   //  void addInter(opencover::coInteractor *inter);
   //  void updateInteractor();
   //  void setCallback(const std::function<void(const ColorMap &)> &f);
    void setMinBounds(float min, float max);
    void setMaxBounds(float min, float max);
    void setMaxNumSteps(int maxSteps);
};

// ColorBar for COVISE/Vistle plugin based on coInteractor
class PLUGIN_UTILEXPORT CoviseColorBar: public ColorBar
{
public:
   CoviseColorBar(ui::Group *menu);
   void addInter(opencover::coInteractor *inter);
   void updateInteractor();
   ~CoviseColorBar();
   void updateFromAttribute(const char *attrib);
   static ColorMap parseAttribute(const char *attrib);
private:
   ui::Action *execute_ = nullptr;
   opencover::coInteractor *inter_ = nullptr;
   void updateGui();
};

// Colorbar for OpenCOVER plugins with direct access to the colorMap
class PLUGIN_UTILEXPORT CoverColorBar: public ColorBar
{
public:
   CoverColorBar(ui::Group *menu);

   void setCallback(const std::function<void(const ColorMap &)> &f);
   void setMinMax(float min, float max, bool autoBounds = true);
   void setSteps(int steps);
   void setSpecies(const std::string &species);
   void setUnit(const std::string &unit);
   const ColorMap &colorMap() const;
   void setColorMap(const std::string& name);
private:
   opencover::ui::SelectionList *m_selector;
   std::function<void(const ColorMap &)> m_callback;

};

}
#endif
