/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PICKSPHERE_PLUGIN_H
#define PICKSPHERE_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/coInteractor.h>

#include <cover/ui/Owner.h>

namespace opencover {
namespace ui {
class Menu;
class Slider;
class Label;
class Action;
class Button;
}
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
#include <map>

using namespace opencover;

class PickSpherePlugin : public coVRPlugin, public ui::Owner
{
public:
    PickSpherePlugin();
    virtual ~PickSpherePlugin();
    void newInteractor(const RenderObject *container, coInteractor *inter);
    void clearTempSpheres();
    void removeObject(const char *objName, bool r);
    void preFrame();
    void addObject(const RenderObject * container, osg::Group *, const RenderObject *, const RenderObject *, const RenderObject *, const RenderObject *);

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
    int x_dim, y_dim, z_dim, Min, Max;         						
    
    // User interface
    void createSubmenu();
    void deleteSubmenu();

    ui::Menu *pickSphereSubmenu = nullptr;
    ui::Slider *startSlider=nullptr, *stopSlider=nullptr;
    ui::Slider *opacityPoti=nullptr;
    ui::Slider *scalePoti=nullptr;
    ui::Button *showTraceCheckbox=nullptr, *regardInterruptCheckbox=nullptr;
    ui::Button *singlePickCheckbox=nullptr, *multiplePickCheckbox=nullptr;
    ui::Button *attachViewerCheckbox=nullptr;
    ui::Action *executeButton=nullptr, *clearSelectionButton=nullptr, *clearPointButton=nullptr;
    ui::Label *particleString=nullptr;
    ui::Slider *x_dimSlider=nullptr, *y_dimSlider=nullptr, *z_dimSlider=nullptr;

    static BoxSelection *boxSelection;
    static void selectWithBox();

    void getSphereData(const covise::coDoSpheres *);
    char *sphereNames;
    const char *startParamName, *stopParamName, *particlesParamName, *UnsortedParticlesParamName, *regardInterruptParamName, *showTraceParamName, *animateViewerParamName, *animateLookAtParamName;
    const char *x_dimParamName, *y_dimParamName, *z_dimParamName;			
    void setParticleStringLabel();

    struct strCmp
    {
        bool operator()(const char *s1, const char *s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };
    std::map<const char *, int, strCmp> addedSphereNames;

    static float s_scale;
};

#endif
