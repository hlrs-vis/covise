/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RecordPath_PLUGIN_H
#define _RecordPath_PLUGIN_H
/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>

using namespace covise;
using namespace opencover;

class PCLPlugin : public coVRPlugin, public coTUIListener
{
public:
    PCLPlugin();
    virtual ~PCLPlugin();
    static PCLPlugin *instance();
    bool init();

    int loadPCD(const char *filename, osg::Group *loadParent);
    static int sloadPCD(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadPCD(const char *filename, const char *covise_key);
    int loadOCT(const char *filename, osg::Group *loadParent);
    static int sloadOCT(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadOCT(const char *filename, const char *covise_key);

    // this will be called in PreFrame
    void preFrame();

private:
    static PCLPlugin *thePlugin;
};
#endif
