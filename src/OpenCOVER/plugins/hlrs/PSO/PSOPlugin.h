/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PSO_PLUGIN_H
#define _PSO_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: PSO Plugin                                                  **
 **                                                                          **
 **                                                                          **
 ** Author: F.Seybold, U.Woessner		                                      **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include "OSGPSOParticle.h"
#include "OSGPSOParticleOnResponseSurface.h"
using namespace covise;
using namespace opencover;

class OSGResponseSurface;

class PSOPlugin : public coVRPlugin, public coTUIListener
{
public:
    PSOPlugin();
    ~PSOPlugin();

    // Initialize Particle-objects
    OSGPSOParticleOnResponseSurface **par2D;
    OSGPSOParticle **par3D;
    // this will be called in PreFrame
    void preFrame();
    double oldTime;

    OSGResponseSurface *surfaceGeode;

    bool init();

private:
    void initPSO();
    void destroyPSO();

    void PSOCycle2D();
    void PSOCycle3D();

    static double quadratic(double *x);
    static double noisy(double *x);

    coTUITab *PSOTab;
    //coTUIEditFloatField *blockAngle;
    //coTUILabel *blockAngleLabel;
    coTUIButton *resetButton;
    coTUIButton *startButton;
    coTUIButton *stopButton;
    coTUIComboBox *functionComboBox;
    coTUIToggleButton *inertiaFunctionToggleButton;
    coTUIToggleButton *crazyFunctionToggleButton;
    coTUIToggleButton *odsFunctionToggleButton;
    void tabletPressEvent(coTUIElement *tUIItem);

    bool isSetup;
    bool isRunning;

    bool applyInertiaUpdate;
    bool applyCrazinessOperator;
    bool applyODS;

    int npar, nvar;
};
#endif
