/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NurbsSurface_PLUGIN_H
#define _NurbsSurface_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: NurbsSurface OpenCOVER Plugin (draws a NurbsSurface)        **
 **                                                                          **
 **                                                                          **
 ** Author: F.Karle/ K.Ahmann                                                **
 **                                                                          **
 ** History:                                                                 **
 ** December 2017  v1                                                        **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

#include <osg/Geode>
#include <cover/coVRCommunication.h>
#include <net/message.h>


#include <string>
#include <cover/ui/Owner.h>

#include "alglib/stdafx.h"
#include "alglib/interpolation.h"




namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
namespace ui {
class Slider;
}
}

using namespace opencover;
using namespace alglib;

class NurbsSurface : public coVRPlugin, public ui::Owner
{
public:
    NurbsSurface();
    ~NurbsSurface();
    bool init();
    virtual bool destroy();
    void message(int toWhom, int type, int len, const void *buf); ///< handle incoming messages
    int getorder_U();
    void setorder_U(int order_U);
    void computeSurface();
    alglib::barycentricinterpolant edge(std::vector<osg::Vec3> all_points, int local_x, int local_y, int change);

private:
    osg::ref_ptr<osg::Geode> geode;


    void saveFile(const std::string &fileName);

    ui::Menu *NurbsSurfaceMenu; //< menu for NurbsSurface Plugin
    ui::Action *saveButton_;

    ui::Slider *orderUSlider=nullptr;
    ui::Slider *orderVSlider=nullptr;
    
    int order_U = 2;
    int order_V = 2;
    
    std::string fileName = "test.obj";

    const int num_points_u = 3; // number of points in the u parameter direction
        const int num_points_v = 3; // number of points in the v parameter direction

        double points[27] = 
        {
            0, 0.005, 0,      0.03, 0.03, 0,      0.1, 0.05, 0,      
            -0.001, 0, 0,      0.03, 0, 0.02,    0.1, 0, 0.05,      
            0, -0.005, 0,      0.03, -0.03, 0,     0.1, -0.05, 0,      
    };

        double u_par[3] = {0, 1, 2}; // point parametrization in u-direction
        double v_par[3] = {0, 1, 2}; // point parametrization in v-direction

        const int dim = 3; // dimension of the space we are working in

        const int num_surf = 1;

        int numberOfAllPoints;
        float maximum_x;
        float minimum_x;
        float maximum_y;
        float minimum_y;

        void initUI();
};
#endif

