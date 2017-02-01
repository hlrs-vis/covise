/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PICKSPHERE_PLUGIN_H
#define PICKSPHERE_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coSliderMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coLabelMenuItem;
class coPotiMenuItem;
}

namespace covise
{
class coDoSpheres;
}

namespace opencover
{
class BoxSelection;
}
#include "PickSphereInteractor.h"
#include "SphereData.h"

using namespace vrui;
using namespace opencover;

class PickSpherePlugin : public coVRPlugin, public coMenuListener
{
public:
    PickSpherePlugin();
    virtual ~PickSpherePlugin();
    void newInteractor(const RenderObject *container, coInteractor *inter);
    void clearTempSpheres();
    void removeObject(const char *objName, bool r);
    void preFrame();
    void addObject(const RenderObject * container, osg::Group *, const RenderObject *, const RenderObject *, const RenderObject *, const RenderObject *);
    coMenuItem *getMenuButton(const std::string &buttonName);

    static float getScale()
    {
        return s_scale;
    }
    coInteractor *getInteractor()
    {
        return inter;
    }

private:
    float *xpos, *ypos, *zpos, *radii;
    int *dummy, maxTimestep;
    int start, stop, min, max;
    std::vector<SphereData *> spheres;
    bool firsttime;
    static PickSphereInteractor *s_pickSphereInteractor;
    coInteractor *inter;
    int animateViewer;
    int animateViewerNumValues;
    char **animateViewerValueNames;
    osg::Vec3 animLookAt;
    std::string traceName;

    // User interface
    void createSubmenu();
    void deleteSubmenu();
    coSubMenuItem *pinboardButton;
    coRowMenu *pickSphereSubmenu;
    coSliderMenuItem *startSlider, *stopSlider;
    coPotiMenuItem *opacityPoti;
    coPotiMenuItem *scalePoti;
    coCheckboxMenuItem *showTraceCheckbox, *regardInterruptCheckbox;
    coCheckboxMenuItem *singlePickCheckbox, *multiplePickCheckbox;
    coCheckboxMenuItem *attachViewerCheckbox;
    coButtonMenuItem *executeButton, *clearSelectionButton;
    coLabelMenuItem *particleString;

    static BoxSelection *boxSelection;
    static void selectWithBox();

    void getSphereData(const covise::coDoSpheres *);
    char *sphereNames;
    const char *startParamName, *stopParamName, *particlesParamName, *regardInterruptParamName, *showTraceParamName, *animateViewerParamName, *animateLookAtParamName;
    void menuEvent(coMenuItem *);
    void setParticleStringLabel();
    void menuReleaseEvent(coMenuItem *);
    struct strCmp
    {
        bool operator()(const char *s1, const char *s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };
    map<const char *, int, strCmp> addedSphereNames;

    static float s_scale;
};

#endif
