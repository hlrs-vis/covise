/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "TemplatePlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

using namespace opencover;

TemplatePlugin::TemplatePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "TemplatePlugin::TemplatePlugin\n");
}

// this is called if the plugin is removed at runtime
TemplatePlugin::~TemplatePlugin()
{
    fprintf(stderr, "TemplatePlugin::~TemplatePlugin\n");
}

// here we get the size and the current center of the cube
void
TemplatePlugin::newInteractor(const RenderObject *container, coInteractor *i)
{
    (void)container;
    (void)i;
    fprintf(stderr, "TemplatePlugin::newInteractor\n");
}

void TemplatePlugin::addObject(const RenderObject *container,
                               osg::Group *root,
                               const RenderObject *obj, const RenderObject *normObj,
                               const RenderObject *colorObj, const RenderObject *texObj)
{
    (void)container;
    (void)obj;
    (void)normObj;
    (void)colorObj;
    (void)texObj;
    (void)root;
    fprintf(stderr, "TemplatePlugin::addObject\n");
}

void
TemplatePlugin::removeObject(const char *objName, bool replace)
{
    (void)objName;
    (void)replace;
    fprintf(stderr, "TemplatePlugin::removeObject\n");
}

void
TemplatePlugin::preFrame()
{
}

COVERPLUGIN(TemplatePlugin)
