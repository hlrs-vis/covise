/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TEMPLATE_PLUGIN_H
#define TEMPLATE_PLUGIN_H
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

class TemplatePlugin : public opencover::coVRPlugin
{
public:
    TemplatePlugin();
    ~TemplatePlugin();

    // this will be called in PreFrame
    void preFrame() override;

    // this will be called if an object with feedback arrives
    void newInteractor(const opencover::RenderObject *container, opencover::coInteractor *i) override;

    // this will be called if a COVISE object arrives
    void addObject(const opencover::RenderObject *container,
                   osg::Group *root,
                   const opencover::RenderObject *obj, const opencover::RenderObject *normObj,
                   const opencover::RenderObject *colorObj, const opencover::RenderObject *texObj) override;

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace) override;

private:
};
#endif
