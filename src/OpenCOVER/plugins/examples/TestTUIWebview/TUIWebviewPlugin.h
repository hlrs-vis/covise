/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUIWEBVIEWPLUGIN_PLUGIN_H
#define TUIWEBVIEWPLUGIN_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>
//#include <tabletUI/TUIWebview.h>
#include "cover/coTabletUI.h"


class WebviewPlugin : public opencover::coVRPlugin
{
public:
    WebviewPlugin();
    ~WebviewPlugin();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(opencover::RenderObject *container, opencover::coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(opencover::RenderObject *container,
                   opencover::RenderObject *obj, opencover::RenderObject *normObj,
                   opencover::RenderObject *colorObj, opencover::RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

    bool init();

    opencover::coTUITab* WebviewTab;
    opencover::coTUIWebview* Webview;
    
private:
};
#endif
