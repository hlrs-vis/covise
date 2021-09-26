/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _IFC_PLUGIN_H
#define _IFC_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: IFC Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

class PLUGINEXPORT ListenerCover;
class PLUGINEXPORT ViewerOsg;
class VrmlScene;
class PLUGINEXPORT SystemCover;
#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>
#include <util/DLinkList.h>


class IfcPlusPlusSystem;

class PLUGINEXPORT IFCPlugin : public coVRPlugin
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;

public:
    static IFCPlugin *plugin;

    IFCPlugin();
    virtual ~IFCPlugin();

    static int loadIFC(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadIFC(const char *filename, const char *ck = "");

    int loadFile(const std::string& fileName, osg::Group* parent);
    int unloadFile(const std::string& fileName);

    IfcPlusPlusSystem* m_system;

    // this will be called in PreFrame
    virtual void preFrame();
    virtual bool init();

    osg::ref_ptr<osg::MatrixTransform> IFCRoot;
    osg::ref_ptr<osg::Switch> IFCSwitch;
    std::list< osg::ref_ptr<osg::Switch>> switches;

private:
};

#endif
