/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: Streetview Plugin				                             **
 **                                                                          **
 **                                                                          **
 ** Author: M.Guedey		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "StreetView.h"

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <osgDB/ReadFile>

#include <osg/Geometry>
#include <osg/MatrixTransform>

using namespace opencover;

StreetView::StreetView()
{
}

bool StreetView::init()
{
   fprintf(stderr, "StreetView::init\n");

   osg::Geode* viereck = new osg::Geode();
   osg::Geometry* viereckGeometry = new osg::Geometry();
   osg::Vec3Array* viereckVertices = new osg::Vec3Array;
   viereckVertices->reserve(4);
   viereckVertices->push_back(osg::Vec3(0,  0,  0));
   viereckVertices->push_back(osg::Vec3(100,0,  0));
   viereckVertices->push_back(osg::Vec3(100,0,100));
   viereckVertices->push_back(osg::Vec3(0,  0,100));
   viereckGeometry->setVertexArray(viereckVertices);
   osg::DrawElementsUShort* viereckDraw = 
      new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4);
   viereckDraw->push_back(0);
   viereckDraw->push_back(1);
   viereckDraw->push_back(2);
   viereckDraw->push_back(3);
   viereckGeometry->addPrimitiveSet(viereckDraw);
   viereck->addDrawable(viereckGeometry);
   
   osg::Vec3Array* viereckNormals = new osg::Vec3Array;
   viereckNormals->reserve(4);
   viereckNormals->push_back(osg::Vec3(0,  -1,  0));
   viereckNormals->push_back(osg::Vec3(0,  -1,  0));
   viereckNormals->push_back(osg::Vec3(0,  -1,  0));
   viereckNormals->push_back(osg::Vec3(0,  -1,  0));
   viereckGeometry->setNormalArray(viereckNormals);
   viereckGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

   osg::Vec2Array* viereckTexcoords = new osg::Vec2Array(4);
   (*viereckTexcoords)[0].set(0.0f,0.0f);
   (*viereckTexcoords)[1].set(1.0f,0.0f);
   (*viereckTexcoords)[2].set(1.0f,1.0f);
   (*viereckTexcoords)[3].set(0.0f,1.0f); 
   viereckGeometry->setTexCoordArray(0,viereckTexcoords);

   // load texture
   std::string pathTexture("\\\\VISFS1\\raid\\share\\projects\\reallabor\\Herrenberg\\Daten\\vonHerrenberg\\Panorama\\P09299_Herrenberg\\EBF");
   osg::Texture2D* viereckTexture = new osg::Texture2D;
   viereckTexture->setDataVariance(osg::Object::DYNAMIC); 
   osg::Image* viereckImage = osgDB::readImageFile(pathTexture+"\\081510001__\\08\\G\\11502118909\\146_149_\\SMA000019.jpg");
   if (!viereckImage)
   {
	  fprintf(stderr, "Couldn't find texture, quiting.\n");
      return -1;
   }
   viereckTexture->setImage(viereckImage);

   osg::StateSet* stateOne = new osg::StateSet();
   stateOne->setTextureAttributeAndModes
      (0,viereckTexture,osg::StateAttribute::ON); 
   viereck->setStateSet(stateOne);

   // root node
   viereckMatrixTransform = new osg::MatrixTransform;
   viereckMatrixTransform->setName("Viereck MatrixTransform");
   viereckMatrixTransform->setMatrix(osg::Matrix::scale(150, 100, 100));
   viereckMatrixTransform->addChild(viereck);

   // add root node to cover scenegraph
   cover->getObjectsRoot()->addChild(viereckMatrixTransform);

   return true;
}

// this is called if the plugin is removed at runtime
StreetView::~StreetView()
{
    fprintf(stderr, "StreetView::~StreetView\n");
	cover->getObjectsRoot()->removeChild(viereckMatrixTransform);
}

void
StreetView::preFrame()
{
}

COVERPLUGIN(StreetView)
