/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SELECT_NODE_PLUGIN_H
#define _SELECT_NODE_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: SelectNode OpenCOVER Plugin selects vertices of a mesh      **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>

class SelectNode : public coVRPlugin
{
public:
    SelectNode();
    virtual ~SelectNode();

    virtual bool destroy();
    virtual bool init();
    virtual void preFrame();

private:
    osg::StateSet *createRedStateSet();

    osg::ref_ptr<osg::Node> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::MatrixTransform> transformSphere; ///< move and scale the sphere
};
#endif
