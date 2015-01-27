/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: ParametricCurves plugin                                   **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** header file                                                            **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _PARAMETRIC_CURVES_PLUGIN_H
#define _PARAMETRIC_CURVES_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <osg/Vec3>
#include <osg/Array>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>

class HfT_osg_Parametric_Surface;
class HfT_osg_Sphere;
class HfT_osg_Plane;
class HfT_osg_MobiusStrip;
class HfT_osg_Animation;

using namespace osg;
using namespace vrui;
using namespace opencover;

class ParametricCurves : public coVRPlugin, public coMenuListener
{
public:
    /****** variables ******/
    /*
   * Static member variable as pointer for the plugin.
   */
    static ParametricCurves *plugin;

    /****** constructors and destructors ******/
    /*
   * Constructor
   * Creates a ParametricCurves object.
   * Initializes the member variables with NULL.
   */
    ParametricCurves();

    /*
   * Destructor
   * Destructs a ParametricCurves object.
   * Cleanup of the pointers to the surfaces.
   * Automatical garbage collection.
   */
    virtual ~ParametricCurves();

    /****** methods ******/
    /*
   * Initializes the plugin.
   * Sets the number of presentation steps.
   * Manages the creation of the surfaces and
   * creates the scene graph tree hierarchy.
   * Creates the slider menu to move the u and v
   * parameter lines.
   *
   * return:       bool
   *               true if the initialization
   *               is valid
   */
    virtual bool init();

    /*
   * Defines which modifications will be
   * done before the next rendering step.
   *
   * return:       void
   */
    void preFrame();

    // no access from outside
private:
    /****** variables ******/
    /*
   * Counter variable to store the current presentation step
   */
    int m_presentationStepCounter;

    /*
   * Counter variable to store the number of presentation steps
   */
    int m_numPresentationSteps;

    /*
   * Variable to store the current presentation step number
   */
    int m_presentationStep;

    /*
   * Variable to store the current u-slider value
   */
    double m_sliderValueU;

    /*
   * Variable to store the current v-slider value
   */
    double m_sliderValueV;

    /*
   * Variable to store the radian of the animation sphere
   */
    double m_animSphereRadian;

    /*
   * Ref pointer template of a switch node, which
   * represents the root node in the created tree.
   * Offers the user to switch between the visibility
   * of the different surfaces.
   */
    ref_ptr<Switch> m_rpRootNodeSwitch;

    /*
   * Pointer to a sphere
   */
    HfT_osg_Sphere *m_pSphere;

    /*
   * Pointer to a second sphere(only for visualisation)
   */
    HfT_osg_Sphere *m_pSphereSecond;

    /*
   * Pointer to a plane
   */
    HfT_osg_Plane *m_pPlane;

    /*
   * Pointer to a mobius strip
   */
    HfT_osg_MobiusStrip *m_pMobius;

    /*
   * Pointer to an animated sphere
   */
    HfT_osg_Animation *m_pAnimation;

    /*
    * Pointer to an animated sphere with an reverse path
    */
    HfT_osg_Animation *m_pAnimationRev;

    /*
    * Pointer to the menu
    */
    coRowMenu *m_pObjectMenu;

    /*
    * Pointer to the slider for the u parameter
    */
    coSliderMenuItem *m_pSliderMenuU;

    /*
   * Pointer to the slider for the v parameter
   */
    coSliderMenuItem *m_pSliderMenuV;

    /****** methods ******/
    /*
   * Calls the constructor of each surface and
   * initializes each surface with standard values
   *
   * return:       void
   */
    void initializeSurfaces();

    /*
   * Creates the slider menu for the directrices
   *
   * return:       void
   */
    void createMenu();

    /*
   * Calculates the presenatation step number with modulo.
   * Defines the visualisation for each presentation step.
   * Initiates a recalculation of the showing object for each step.
   *
   * return:       void
   */
    void changePresentationStep();

    /*
   * Controls the messages which are sent
   * from the toolbar by pushing the toolbar buttons.
   * A counter stores the current presentation step.
   *
   * Parameters:       const char *msg
   *                   pointer to char for the message
   *
   * return:       void
   */
    void guiToRenderMsg(const char *msg);

    /*
    * Controls which slider button in the slider menu is moved.
    * Dependent on the current presentation step, the correct
    * directrix is shown.
    *
    * Parameters:       coMenuItem *iMenuItem
    *                   chosed menu button
    *
    * return:       void
    */
    void menuEvent(coMenuItem *iMenuItem);
    void setMenuVisible(bool visible);
};

#endif
