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
#include <cover/coTabletUI.h>

#include <OpenVRUI/coMenu.h>

#include <osg/Geode>
#include <cover/coVRCommunication.h>
#include <net/message.h>


#include <string>

namespace vrui
{
class coButtonMenuItem;
}
namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
}

using namespace vrui;
using namespace opencover;

class NurbsSurface : public coVRPlugin,
                     public coMenuListener,
                     public coTUIListener   
{
public:
    NurbsSurface();
    ~NurbsSurface();
    bool init();
    virtual bool destroy();
    void message(int toWhom, int type, int len, const void *buf); ///< handle incoming messages

private:
    osg::ref_ptr<osg::Geode> geode;

    mutable bool doSave;

    bool doInit;

    coButtonMenuItem *SaveButton;

    coTUIButton *tuiSaveButton;
    coTUITab *tuiSaveTab;
    coTUILabel *tuiFileNameLabel;
    coTUIEditField *tuiFileName;
    coTUILabel *tuiSavedFileLabel;
    coTUILabel *tuiSavedFile; 

    //void prepareSafe();

   // virtual void menuEvent(coMenuItem *);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);   

    void initUI();
};
#endif

