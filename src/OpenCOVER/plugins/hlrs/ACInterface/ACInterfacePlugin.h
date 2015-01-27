/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ACINTERFACE_PLUGIN_H
#define _ACINTERFACE_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2007/8 HLRS **
 **                                                                           **
 ** Description: ACInterfacePlugin									            **
 **																		                     **
 **                                                                           **
 ** Author: Mario Baalcke	                                                   **
 **                                                                           **
 **                                                                           **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRSelectionManager.h>

class OpenCOVERProxy;

class ACInterfacePlugin : public coVRPlugin, public coSelectionListener
{
public:
    ACInterfacePlugin();
    virtual ~ACInterfacePlugin();
    bool init();

    virtual bool selectionChanged();
    virtual bool pickedObjChanged();

    void preFrame();

    static ACInterfacePlugin *plugin;

    void message(int type, int len, const void *buf);

private:
    coVRSelectionManager *selectionManager;

    OpenCOVERProxy *service;
};
#endif
