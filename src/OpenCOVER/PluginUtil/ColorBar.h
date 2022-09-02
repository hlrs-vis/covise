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
}

class PLUGIN_UTILEXPORT ColorBar: public ui::Owner
{
private:
    vrui::vruiMatrix *floatingMat_ = nullptr;
    coColorBar *colorbar_ = nullptr, *hudbar_ = nullptr;
    ui::Menu *colorsMenu_ = nullptr;
    std::string title_;
    std::string name_;
    std::string species_;
    ui::SpecialElement *uiColorBar_ = nullptr;
    ui::Slider *minSlider_ = nullptr;
    ui::Slider *maxSlider_ = nullptr;
    ui::Slider *stepSlider_ = nullptr;
    ui::Button *autoScale_ = nullptr;
    ui::Action *execute_ = nullptr;
    ui::Slider *center_ = nullptr;
    ui::Slider *compress_ = nullptr;
    ui::Slider *insetCenter_ = nullptr;
    ui::Slider *insetWidth_ = nullptr;
    ui::Slider *opacityFactor_ = nullptr;
    ui::Button *show_ = nullptr;

    opencover::coInteractor *inter_ = nullptr;

    void updateTitle();

    float min = 0.0;
    float max = 1.0;
    int numColors = 0;
    std::vector<float> r, g, b, a;

public:

    /** constructor when the colorbar is not to be opened from the pinboard
       *  create create containers, texture and labels
       *  @param colorsButton the button that opens the colorbar
       *  @param moduleMenu the coRowMenu used instead of the pinboard
       *  @param species data species name, currently not displayed
       *  @param min data minimum
       *  @param max data maximum
       *  @param numColors number of different colors in colorbar
       *  @param r red colors
       *  @param g green colors
       *  @param b blue colors
       *  @param a red colors
       */
    ColorBar(ui::Menu *menu);

    /// destructor
    ~ColorBar();

    bool hudVisible() const;
    void setHudPosition(osg::Vec3 pos, osg::Vec3 hpr, float size);

    /** colorbar update
       *  @param species title bar content
       *  @param min data minimum
       *  @param max data maximum
       *  @param numColors number of different colors in colorbar
       *  @param r red colors
       *  @param g green colors
       *  @param b blue colors
       *  @param a red colors
       */
    void update(const std::string &species, float min, float max, int numColors, const float *r, const float *g, const float *b, const float *a);

    /** set name */
    void setName(const char *name);
    void show(bool state);
    /** get name
     *  @return name the name of the colorbar, identical with module name, eg, g, Colors_1
     */
    const char *getName();

    /** parseAttrib
       * @param attrib COLORMAP attribute
       * @param species: get species (the client should delete this pointer)
       * @param min: colormap minimum
       * @param max: colormap maximum
       * @param numColors: number of colors
       * @param r: red (the client should delete this pointer)
       * @param g: green (the client should delete this pointer)
       * @param b: blue (the client should delete this pointer)
       * @param a: alpha (the client should delete this pointer)
       */
    static void parseAttrib(const char *attrib, std::string &species,
                            float &min, float &max, int &numColors,
                            std::vector<float> &r, std::vector<float> &g, std::vector<float> &b, std::vector<float> &a);

    void parseAttrib(const char *attrib);
    void setVisible(bool);
    bool isVisible();

    void addInter(opencover::coInteractor *inter);
    void updateInteractor();
};
}
#endif
