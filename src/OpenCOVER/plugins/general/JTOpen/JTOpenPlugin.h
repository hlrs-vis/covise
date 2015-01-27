/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _JTOpen_PLUGIN_H
#define _JTOpen_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: JTOpen Plugin (does nothing)                              **
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
class JtkHierarchy;
class JtkClientData;
#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>
#include <util/DLinkList.h>
class JtkShape;

typedef std::list<osg::ref_ptr<osg::Group> > ParentList;
class PLUGINEXPORT JTOpenPlugin : public coVRPlugin
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;

public:
    static JTOpenPlugin *plugin;

    JTOpenPlugin();
    virtual ~JTOpenPlugin();

    static int loadJT(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadJT(const char *filename, const char *ck = "");

    osg::Node *createShape(JtkShape *partShape, const char *objName);
    static int myPreactionCB(JtkHierarchy *CurrNode, int level, JtkClientData *cd);
    int PreAction(JtkHierarchy *CurrNode, int level, JtkClientData *);
    static int myPostactionCB(JtkHierarchy *CurrNode, int level, JtkClientData *cd);
    int PostAction(JtkHierarchy *CurrNode, int level, JtkClientData *);
    // this will be called in PreFrame
    virtual void preFrame();
    virtual bool init();

    void setMaterial(osg::Node *, JtkHierarchy *CurrNode);
    osg::Group *createGroup(JtkHierarchy *CurrNode);
    osg::ref_ptr<osg::Group> currentGroup;
    ParentList Parents;
    osg::ref_ptr<osg::Group> firstGroup;

private:
    bool scaleGeometry;
    float lodScale;
};

#endif
