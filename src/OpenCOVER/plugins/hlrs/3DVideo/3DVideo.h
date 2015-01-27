/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Video3D_PLUGIN_H
#define _Video3D_PLUGIN_H
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
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <osg/Drawable>
#include <osg/Geode>

class Video3DNode : public osg::Drawable
{
private:
public:
    Video3DNode();
    virtual ~Video3DNode();
    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    /** Clone the type of an object, with Object* return type.
	Must be defined by derived classes.*/
    virtual osg::Object *cloneType() const;

    /** Clone the an object, with Object* return type.
	Must be defined by derived classes.*/
    virtual osg::Object *clone(const osg::CopyOp &) const;
};
class Video3D
{
public:
    Video3D();
    ~Video3D();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void feedback(coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *parent,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

private:
    Video3DNode *videoNode;
    osg::Geode *geodevideo;
};
#endif
