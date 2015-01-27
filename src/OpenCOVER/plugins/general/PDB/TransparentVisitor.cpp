/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <cover/coVRPluginSupport.h>
#include "TransparentVisitor.h"

using namespace opencover;

TransparentVisitor::TransparentVisitor()
    : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
    _alpha = 0;
    _transparent = false;
    _transparent_mode = false;
    _scale = 1;
}

void TransparentVisitor::setDistance(float dist)
{
    _distance = dist;
}

void TransparentVisitor::setScale(float scale)
{
    _scale = scale;
}

void TransparentVisitor::setArea(float area)
{
    _area = area;
}

void TransparentVisitor::enableTransparency(bool trans)
{
    _transparent_mode = trans;
    _alpha = 0;
    //_transparent = true;
}

// need to adjust the scale (have independent scale options for each structure)
float TransparentVisitor::getDistance(osg::Vec3 position1, osg::Vec3 position2)
{

    return (float)((position1 - position2).length() / _scale);
}

void TransparentVisitor::apply(osg::Geode &node)
{
    if (_transparent_mode)
    {
        // calculate alpha
        osg::Vec3 viewerPosWorld = cover->getViewerMat().getTrans();
        //osg::Vec3 structurePosWorld;
        osg::Vec3 structurePosR2;

        osg::BoundingBox box = node.getBoundingBox();
        structurePosR2 = box.center(); // * cover->getBaseMat();

        currentnode = &node; // get the pointer to the current node
        while (currentnode->getNumParents() > 0)
        {
            currentnode = currentnode->getParent(0);

            osg::MatrixTransform *matrix = dynamic_cast<osg::MatrixTransform *>(currentnode);
            if (matrix)
            {
                structurePosR2 = structurePosR2 * matrix->getMatrix();
            }
        }

        // calculate alpha and bin
        calculateAlphaAndBin(getDistance(structurePosR2, viewerPosWorld));
    }

    // adjust the drawables
    for (int i = 0; i < node.getNumDrawables(); i++)
    {
        drawable = dynamic_cast<osg::Drawable *>(node.getDrawable(i));
        if (drawable->asGeometry())
        {
            osg::Geometry *geo = drawable->asGeometry();

            // append transparent stateset
            stateset = geo->getOrCreateStateSet();

            osg::Vec4Array *color = (osg::Vec4Array *)geo->getColorArray();

            (*color)[0].set((*color)[0][0], (*color)[0][1], (*color)[0][2], _alpha);

            geo->setColorArray(color);

            if (_transparent)
            {
                stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
                stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
                stateset->setMode(GL_BLEND, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
                stateset->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
            }
            else
            {
                stateset->setRenderingHint(osg::StateSet::OPAQUE_BIN);
                stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                stateset->setMode(GL_BLEND, osg::StateAttribute::OFF);
                stateset->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
            }
            /*
	  	// append transparent stateset
	  	stateset = geo->getOrCreateStateSet();
		stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

	  	// get the material if it exists
	  	material = dynamic_cast<osg::Material*>(stateset->getAttribute(osg::StateAttribute::MATERIAL));
	  	if(material)
	  	{
		  material->setTransparency(osg::Material::FRONT_AND_BACK, _alpha);

	  	  if(_transparent)
	  	  {
       		    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
       	  	    //stateset->setMode(GL_BLEND, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
		    //stateset->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
		    stateset->setAttribute(material, osg::StateAttribute::OVERRIDE);
		    cerr << "ALPHA IS TRANSPARENT " <<  _alpha << endl;
	  	  }
	  	  else
	  	  {
		    stateset->setRenderingHint(osg::StateSet::OPAQUE_BIN);
       	  	    //stateset->setMode(GL_BLEND, osg::StateAttribute::OFF);
		    //stateset->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
		    stateset->setAttribute(material, osg::StateAttribute::OFF);
		    cerr << "ALPHA IS NON TRANSPARENT " <<  _alpha << endl;
	  	  }
		}
		*/
        }
    }

    /*
	        // append transparent stateset
	        stateset = node.getOrCreateStateSet();
	   
	        if(_transparent)
		{
           	
	  		// get the material if it exists
	  		material = dynamic_cast<osg::Material*>(stateset->getAttribute(osg::StateAttribute::MATERIAL));
	  		if(!material)
		  		material = new osg::Material();
		
		    	material->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0, 0.0, 0.0,1.0));
		    	material->setTransparency(osg::Material::FRONT_AND_BACK,  _alpha);
		    	stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
       	  	    	stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
		    	stateset->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
		    	stateset->setAttribute(material, osg::StateAttribute::OVERRIDE);
	  	}
	  	else
	  	{
		 	stateset->removeAttribute(osg::StateAttribute::MATERIAL);
	  	}
	    */
}
