/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef AXIALAR_PLUGIN_H
#define AXIALAR_PLUGIN_H

#include <deque>

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/ARToolKit.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

class AxialARPlugin : public coVRPlugin, public coTUIListener, public coMenuListener
{
public:
    AxialARPlugin();
    virtual ~AxialARPlugin();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void feedback(coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   const char *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, int replace);
    bool idExists(int ID);
    void menuEvent(coMenuItem *);
    virtual void focusEvent(bool focus, coMenu *menu);

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

private:
    float getAngle();
    void updateAndExec();

    ARToolKitMarker *angleMarker;

    coButtonMenuItem *execButton;
    coSubMenuItem *pinboardEntry;
    coRowMenu *arMenu;

    coRowMenu *simulationMenu;
    coTUITab *tab;
    coTUIButton *restartSimulation;

    coInteractor *interactor;

    //float angle;
    std::deque<float> angle;
};
#endif
