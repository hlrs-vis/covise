/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rhino_PLUGIN_H
#define _Rhino_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Rhino Plugin (does nothing)                              **
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

#include <osg/Group>

#include <opennurbs.h>



class PLUGINEXPORT RhinoPlugin : public coVRPlugin
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;

public:
    static RhinoPlugin *plugin;

    RhinoPlugin();
    virtual ~RhinoPlugin();

    static int loadRhino(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadRhino(const char *filename, const char *ck = "");

    int loadFile(const std::string& fileName, osg::Group* parent);
    int unloadFile(const std::string& fileName);


    // this will be called in PreFrame
    virtual bool update();
    virtual bool init();

    osg::ref_ptr<osg::MatrixTransform> RhinoRoot;
    osg::ref_ptr<osg::Switch> RhinoSwitch;
    std::list< osg::ref_ptr<osg::Switch>> switches;

    osg::Node* GetRhinoModel(void);

    void SetEdgePrecision(double thePrecision);
    void SetFacePrecision(double thePrecision);


private:
    void Read3DM(const std::string& theFileName);

    osg::Node* BuildBrep(const ON_Brep* theBrep);

    osg::Node* BuildEdge(const ON_Brep* theBrep);

    osg::Node* BuildWireFrameFace(const ON_BrepFace* theFace);

    osg::Node* BuildShadedFace(const ON_BrepFace* theFace);

    double mEdgePrecision;

    double mFacePrecision;

    ONX_Model mRhinoModel;

    osg::Node* mRhinoNode;

};

#endif
