#include "Picture.h"
#include "Index.h"
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <osgDB/ReadFile>

#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Texture2D>

#include <iostream>

Picture::Picture(xercesc::DOMNode *pictureNode, Index *index_)
{
	latitude = 0.0;
	longitude = 0.0;
	altitude = 0.0;
	heading = 0.0;
	index = index_;
	
	xercesc::DOMNodeList *pictureNodeList = pictureNode->getChildNodes();
	for (int j = 0; j < pictureNodeList->getLength(); ++j)
	{
		xercesc::DOMElement *pictureElement = dynamic_cast<xercesc::DOMElement *>(pictureNodeList->item(j));
		if (!pictureElement)
			continue;
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Filename"), pictureElement->getTagName()))
		{
			fileName = xercesc::XMLString::transcode(pictureElement->getTextContent());
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("LAT"), pictureElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(pictureElement->getTextContent());
			sscanf(tmp, "%lf", &latitude);
		}	
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("LON"), pictureElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(pictureElement->getTextContent());
			sscanf(tmp, "%lf", &longitude);
		}	
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("ALT"), pictureElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(pictureElement->getTextContent());
			sscanf(tmp, "%lf", &altitude);
		}	
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Heading"), pictureElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(pictureElement->getTextContent());
			sscanf(tmp, "%lf", &heading);
		}
	}

}

Picture::~Picture()
{
}

osg::Node *Picture::getPanelNode()
{
	osg::Geode *viereck = new osg::Geode();
	osg::Geometry *viereckGeometry = new osg::Geometry();
	osg::Vec3Array *viereckVertices = new osg::Vec3Array;
	viereckVertices->reserve(4);
	viereckVertices->push_back(osg::Vec3(0,  0,  0));
	viereckVertices->push_back(osg::Vec3(100,0,  0));
	viereckVertices->push_back(osg::Vec3(100,0,100));
	viereckVertices->push_back(osg::Vec3(0,  0,100));
	viereckGeometry->setVertexArray(viereckVertices);
	osg::DrawElementsUShort *viereckDraw = 
		new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4);
	viereckDraw->push_back(0);
	viereckDraw->push_back(1);
	viereckDraw->push_back(2);
	viereckDraw->push_back(3);
	viereckGeometry->addPrimitiveSet(viereckDraw);
	viereck->addDrawable(viereckGeometry);

	osg::Vec3Array *viereckNormals = new osg::Vec3Array;
	viereckNormals->reserve(4);
	viereckNormals->push_back(osg::Vec3(0,  -1,  0));
	viereckNormals->push_back(osg::Vec3(0,  -1,  0));
	viereckNormals->push_back(osg::Vec3(0,  -1,  0));
	viereckNormals->push_back(osg::Vec3(0,  -1,  0));
	viereckGeometry->setNormalArray(viereckNormals);
	viereckGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

	osg::Vec2Array *viereckTexcoords = new osg::Vec2Array(4);
	(*viereckTexcoords)[0].set(0.0f,0.0f);
	(*viereckTexcoords)[1].set(1.0f,0.0f);
	(*viereckTexcoords)[2].set(1.0f,1.0f);
	(*viereckTexcoords)[3].set(0.0f,1.0f); 
	viereckGeometry->setTexCoordArray(0,viereckTexcoords);

	// load texture
	osg::Texture2D* viereckTexture = new osg::Texture2D;
	viereckTexture->setDataVariance(osg::Object::DYNAMIC); 
	osg::Image *viereckImage = osgDB::readImageFile(index->getAbsolutePicturePath()+fileName);
	if (!viereckImage)
	{
		fprintf(stderr, "Image missing!\n");
	}
	else
	{
		viereckTexture->setImage(viereckImage);
	}

	osg::StateSet *stateOne = new osg::StateSet();
	stateOne->setTextureAttributeAndModes(0,viereckTexture,osg::StateAttribute::ON); 
	viereck->setStateSet(stateOne);

	// root node
	viereckMatrixTransform = new osg::MatrixTransform;
	viereckMatrixTransform->setName("Viereck MatrixTransform");
	viereckMatrixTransform->setMatrix(osg::Matrix::scale(150, 100, 100));
	viereckMatrixTransform->addChild(viereck);
	return viereckMatrixTransform;
}