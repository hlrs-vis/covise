/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ThreeDTK_PLUGIN_H
#define _ThreeDTK_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2013 HLRS  **
 **                                                                          **
 ** Description: ThreeDTK Plugin (loads and renders PointCloud)              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Nov-01  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <co3dtkDrawable.h>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>

using namespace opencover;

class ThreeDTK : public coVRPlugin
{
public:
    ThreeDTK();
    ~ThreeDTK();

    // this will be called in PreFrame
    void preFrame();

    void cycleLOD();
    int loadFile(std::string filename);
    int readFrames(std::string dir, int start, int end, bool readInitial, reader_type &type);
    void generateFrames(int start, int end, bool identity);
    void createDisplayLists(bool reduced);

    static int loadFile(const char *name, osg::Group *parent);
    static int unloadFile(const char *name);

private:
    osg::ref_ptr<co3dtkDrawable> drawable;
    osg::ref_ptr<osg::Geode> node;
    osg::ref_ptr<osg::Material> mtl;
    osg::ref_ptr<osg::StateSet> geoState;
};
#endif
