/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLMXML_PLUGIN_H
#define _PLMXML_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: PLMXML Plugin (load plmxml documents)                       **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                             **
 **                                                                          **
 ** History:  					                             **
 ** Nov-01  v1	    				                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRShader.h>
#include "cover/coTabletUI.h"
#include <util/coTabletUIMessages.h>
#include <util/coRestraint.h>
#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>

class PLUGINEXPORT PLMXMLPlugin : public coVRPlugin, public coTUIListener
{
public:
    static PLMXMLPlugin *plugin;

    PLMXMLPlugin();
    ~PLMXMLPlugin();

    bool init();

    static int loadPLMXML(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadPLMXML(const char *filename, const char *ck = "");
    static osg::Group *currentGroup;
    void addNode(osg::Node *, RenderObject *);
    //      virtual void tabletSwitchView(char *nodeName);
private:
};

#endif
