/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// class coColorBar
//
// derived from coMenuItem
// colorbar is a window containing a texture and labels
//
// Initial version: 2002-01, dr
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by Vircinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _CO_COLOR_BAR_H_
#define _CO_COLOR_BAR_H_

#define MAX_LABELS 18

#include <OpenVRUI/coMenuItem.h>
#include <util/coTypes.h>

namespace vrui
{
class coLabel;
class coRowContainer;
class coTexturedBackground;
class coColoredBackground;
}

namespace opencover
{

/** class coColorBar, derived from coMenuItem
 *  colorbar is a window containing a texture and labels
 */
class PLUGIN_UTILEXPORT coColorBar : public vrui::coMenuItem
{
private:
    vrui::coColoredBackground *background_;

    // topspacer and texture in vertical row container
    vrui::coColoredBackground *vspace_;
    vrui::coTexturedBackground *texture_;
    vrui::coRowContainer *textureAndVspace_;

    // hspacers and labels in horiz containers, all labels in vert container
    vrui::coLabel *labels_[MAX_LABELS];
    vrui::coTexturedBackground *hspaces_[MAX_LABELS];
    vrui::coRowContainer *labelAndHspaces_[MAX_LABELS];
    vrui::coRowContainer *allLabels_;

    // horiz conatiner around textureAndVspacer and all labels
    vrui::coRowContainer *textureAndLabels_;

    int numLabels_; // number of labels, max
    float labelValues_[MAX_LABELS]; // numerical values of labels
    char format_str_[32]; // precision of float values

    int numColors_;
    float min_, max_;
    unsigned char *image_, *tickImage_;
    char *name_; // the name of the colors module for example Colors_1

    void makeImage(int numColors, float *r, float *g, float *b, float *a);
    void makeTickImage();
    void makeLabelValues();

public:
    /** constructor
       *  create texture and labels, put them nto containers
       *  @param name the name of the colorbar, identical with module name, eg, g, Colors_1
       *  @param species data species name, currently not displayed
       *  @param min data minimum
       *  @param max data maximum
       *  @param numColors number of different colors in colorbar
       *  @param r red colors
       *  @param g green colors
       *  @param b blue colors
       *  @param a red colors
       */
    coColorBar(const char *name, char *species, float min, float max, int numColors, float *r, float *g, float *b, float *a);

    /// destructor
    ~coColorBar();

    /** colorbar update
       *  @param min data minimum
       *  @param max data maximum
       *  @param numColors number of different colors in colorbar
       *  @param r red colors
       *  @param g green colors
       *  @param b blue colors
       *  @param a red colors
       */
    void update(float min, float max, int numColors, float *r, float *g, float *b, float *a);

    /** get name
       *  @return name the name of the colorbar, identical with module name, eg, g, Colors_1
       */
    const char *getName()
    {
        return name_;
    };

    virtual vrui::coUIElement *getUIElement();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);
};
}

#endif
