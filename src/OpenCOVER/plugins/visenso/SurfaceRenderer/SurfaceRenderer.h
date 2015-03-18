/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: Plugin class for the renderer                             **
 **                                                                        **
 ** header file                                                            **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     01.2011 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef SURFACERENDERER_H_
#define SURFACERENDERER_H_

#include <Python.h>
#include "ParamSurface.h"
//#include "PlaneSurface.h"
// SphereSurface not available#include "SphereSurface.h"
#include "MobiusStrip.h"
//#include "ZylinderSurface.h"
//#include "WendelSurface.h"
//#include "KegelSurface.h"
#include <string>
#include <iostream>

#include <PluginUtil/GenericGuiObject.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/RenderObject.h>

#include <grmsg/coGRGenericParamRegisterMsg.h>
#include <grmsg/coGRGenericParamChangedMsg.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <osg/MatrixTransform>
#include <osg/Switch>
#include <config/CoviseConfig.h>

//class ParamSurface;
//class PlaneSurface;
//class ZylinderSurface;
//class WendelSurface;
//class KegelSurface;

using namespace std;
using namespace osg;

using namespace vrui;
using namespace opencover;

class SurfaceRenderer : public coVRPlugin, public GenericGuiObject, public coMenuListener
{

public:
    //Constructor Destructor
    SurfaceRenderer();
    virtual ~SurfaceRenderer();

    //variables of class
    static SurfaceRenderer *plugin;

    //-----------------inherit----------------------------//
    virtual bool init(); //Initialises
    virtual void preFrame(); //Defines which modifications will be done before the next rendering step
    virtual void guiToRenderMsg(const char *msg); //Called if a message from the GUI is received
    virtual void menuEvent(coMenuItem *menuItem); //Called if menu is used
    coRowMenu *getObjectsMenu()
    {
        return menuItemMenu;
    };

    void update();

    void setActive(bool active);
    bool isActive()
    {
        return active;
    };

    virtual void setCharge(float charge);
    float getCharge()
    {
        return charge;
    };

    virtual void setCharge(int charge);
    int getCharge_()
    {
        return charge_;
    };

    virtual void setString(std::string string_);
    std::string getString()
    {
        return _string;
    };

    void setVisible(char visible);
    // void testingLink();
    //     void addSphere();

protected:
    void createMenu(); //called to create Menu in vr-prepare
    void guiParamChanged(GuiParam *guiParam); //refresh of GUI
    void recreationSurface();

private:
    ref_ptr<Switch> m_rpRootSwitchNode; //root node of the self created scenegraph
    ref_ptr<MatrixTransform> m_rpTransMatrixNode;

    // PlaneSurface	*m_pPlaneSurface;
    //ParamSurface     *m_pSurface; //created surface by the parametrical description

    // SphereSurface not availableSphereSurface  	*m_pSphereSurface;
   ParamSurface *m_Sphere; //created Sphere

    ParamSurface *m_Strip; //created MobiusStrip
    MobiusStrip *m_pMobiusStrip;

    /* WendelSurface	*m_pWendelSurface;
       ZylinderSurface	*m_pZylinderSurface;
       KegelSurface	*m_pKegelSurface;
       ParamSurface	*m_pDiniSurface;
       ParamSurface	*m_pTorusSurface;
       ParamSurface	*m_pSattelSurface;
       ParamSurface	*m_pBoloidSurface;
       ParamSurface	*m_pFlascheSurface;*/

    //  ParamSurface     *m_Surface; //created surface by the parametrical description
    // ParamSurface	*m_Plane;//created Plane
    /*ParamSurface	*m_Wendel; //created Helikoid
       ParamSurface	*m_Zylinder;//created Zylinder
       ParamSurface	*m_Kegel;//created Kegel
       ParamSurface	*m_Dini;//created Dini Surface
       ParamSurface	*m_Torus;//created Torus
       ParamSurface	*m_Sattel;//created Hyperboloid
       ParamSurface	*m_Boloid;//created Paraboloid
       ParamSurface	*m_Flasche;//created Klein Bottle*/
    ParamSurface *m_CreatedCurve; //created own changend Curve

    //---------------Menu in vr-prepare--------------------//
    GuiParamBool *gui_showMenu;
    GuiParamBool *gui_creationMode;
    GuiParamString *gui_stringxRow;
    GuiParamString *gui_stringyRow;
    GuiParamString *gui_stringzRow;
    GuiParamString *gui_stringNormalXRow;
    GuiParamString *gui_stringNormalYRow;
    GuiParamString *gui_stringNormalZRow;
    GuiParamInt *gui_surfaceMode;
    GuiParamInt *gui_patU;
    GuiParamInt *gui_patV;
    GuiParamFloat *gui_lowBoundU;
    GuiParamFloat *gui_upBoundU;
    GuiParamFloat *gui_lowBoundV;
    GuiParamFloat *gui_upBoundV;

    //-------------------Variables------------------------------//
    std::string p_x_row;
    std::string p_y_row;
    std::string p_z_row;
    std::string p_mx_row;
    std::string p_my_row;
    std::string p_mz_row;

    std::string _string;
    std::string string_;

    std::string r;
    int p_surface_mode;

    float p_lowbound_u;
    float p_upbound_u;
    float p_lowbound_v;
    float p_upbound_v;

    bool allclear;
    bool active;

    float charge;
    int charge_;

    int visible_;
    char visible;

    int gui_string;
    int gui_charge_;
    float gui_charge;

    //------------------MenuItems in OpenCover------------------------------//
    coRowMenu *menuItemMenu;

    coButtonMenuItem *menuItemAddSphere;
    coButtonMenuItem *menuItemAddStrip;
    coButtonMenuItem *menuItemAddWendel;
    coButtonMenuItem *menuItemAddZylinder;
    coButtonMenuItem *menuItemAddDini;
    coButtonMenuItem *menuItemAddTorus;
    coButtonMenuItem *menuItemAddSattel;
    coButtonMenuItem *menuItemAddBoloid;
    coButtonMenuItem *menuItemAddKegel;
    coButtonMenuItem *menuItemAddFlasche;
    coButtonMenuItem *menuItemAddLine;
    coButtonMenuItem *menuItemAddPlane;

    //------------Methods---------------------------------
    void addMobius_Strip();

    void getModifications();
    void addWendel_Surface();
    void addZylinder();
    void addDini_Surface();
    void addTorus();
    void addSattelflaeche();
    void addBoloid();
    void addKegel();
    void addFlasche();
    void addPlane();

    void initializeSurfaces();
};

#endif /* SURFACERENDERER_H_ */
