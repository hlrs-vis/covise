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

class PLUGIN_UTILEXPORT ColorBar: public ui::Owner
{
private:
    coColorBar *colorbar_ = nullptr;
    ui::Menu *colorsMenu_ = nullptr;
    std::string title_;
    std::string name_;
    std::string species_;
    ui::Slider *minSlider_ = nullptr;
    ui::Slider *maxSlider_ = nullptr;
    ui::Slider *stepSlider_ = nullptr;
    ui::Button *autoScale_ = nullptr;
    ui::Action *execute_ = nullptr;

    opencover::coInteractor *inter_ = nullptr;

    void updateTitle();

    /// The TabletUI Interface
    opencover::coTUITab *_tab;
    opencover::coTUIEditFloatField *_min;
    opencover::coTUIEditFloatField *_max;
    opencover::coTUIEditIntField *_steps;
    opencover::coTUILabel *_minL;
    opencover::coTUILabel *_maxL;
    opencover::coTUILabel *_stepL;

    opencover::coTUIToggleButton *_autoScale;
    opencover::coTUIButton *_execute;

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
    ColorBar(ui::Menu *menu, char *species, float min, float max, int numColors, float *r, float *g, float *b, float *a);

    /// destructor
    ~ColorBar();

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
    void update(const char *species, float min, float max, int numColors, float *r, float *g, float *b, float *a);

    /** set name */
    void setName(const char *name);

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
    static void parseAttrib(const char *attrib, char *&species,
                            float &min, float &max, int &numColors,
                            float *&r, float *&g, float *&b, float *&a);

    void setVisible(bool);
    bool isVisible();

    void addInter(opencover::coInteractor *inter);
};
}
#endif
