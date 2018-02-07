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
 ** Author: F.Karle/ K.Ahmann	                                             **
 **                                                                          **
 ** History:  			  	                                     **
 ** December 2017  v1		                                             **
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


namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
namespace ui {
class Slider;
}
}

using namespace opencover;

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
//	int getorder_V();
//	void setorder_V(int order_V);


private:
   	osg::ref_ptr<osg::Geode> geode;


	void saveFile(const std::string &fileName);

	ui::Menu *NurbsSurfaceMenu; //< menu for NurbsSurface Plugin
	ui::Action *saveButton_;

	ui::Slider *orderUSlider=nullptr;
	ui::Slider *orderVSlider=nullptr;

	int order_U = 5;
    	int order_V = 5;


    	void initUI();
};
#endif

