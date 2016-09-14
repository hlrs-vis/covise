#include "Picture.h"
#include "Index.h"
#include "Camera.h"

#include <osgDB/ReadFile>

#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Texture2D>

#include <iostream>

Picture::Picture(xercesc::DOMNode *pictureNode, Index *index_)
{
	station = 0;
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
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Buchst"), pictureElement->getTagName()))
		{
			cameraType = xercesc::XMLString::transcode(pictureElement->getTextContent());
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Station"), pictureElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(pictureElement->getTextContent());
			sscanf(tmp, "%d", &station);
		}	
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

	// Zeiger auf Cam

}

Picture::~Picture()
{
}

osg::Node *Picture::getPanelNode()
{
	osg::Geode *panel = new osg::Geode();
	panel->setName("Panel");

	osg::Geometry *panelGeometry = new osg::Geometry();

	osg::Vec3Array *panelVertices = new osg::Vec3Array;
	panelVertices->reserve(4);
	panelVertices->push_back(osg::Vec3(0,  0,  0));
	panelVertices->push_back(osg::Vec3(100,0,  0));
	panelVertices->push_back(osg::Vec3(100,0,100));
	panelVertices->push_back(osg::Vec3(0,  0,100));
	panelGeometry->setVertexArray(panelVertices);

	osg::DrawElementsUShort *panelDraw = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4);
	panelDraw->push_back(0);
	panelDraw->push_back(1);
	panelDraw->push_back(2);
	panelDraw->push_back(3);
	panelGeometry->addPrimitiveSet(panelDraw);
	panel->addDrawable(panelGeometry);

	osg::Vec3Array *panelNormals = new osg::Vec3Array;
	panelNormals->reserve(4);
	panelNormals->push_back(osg::Vec3(0,  -1,  0));
	panelNormals->push_back(osg::Vec3(0,  -1,  0));
	panelNormals->push_back(osg::Vec3(0,  -1,  0));
	panelNormals->push_back(osg::Vec3(0,  -1,  0));
	panelGeometry->setNormalArray(panelNormals);
	panelGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

	osg::Vec2Array *panelTexcoords = new osg::Vec2Array(4);
	(*panelTexcoords)[0].set(0.0f,0.0f);
	(*panelTexcoords)[1].set(1.0f,0.0f);
	(*panelTexcoords)[2].set(1.0f,1.0f);
	(*panelTexcoords)[3].set(0.0f,1.0f); 
	panelGeometry->setTexCoordArray(0,panelTexcoords);

	osg::Texture2D* panelTexture = new osg::Texture2D;
	panelTexture->setDataVariance(osg::Object::DYNAMIC); 
	osg::Image *panelImage = osgDB::readImageFile(index->getAbsolutePicturePath()+fileName);
	if (!panelImage)
	{
		fprintf(stderr, "Image missing!\n");
	}
	else
	{
		panelTexture->setImage(panelImage);
	}

	osg::StateSet *stateOne = new osg::StateSet();
	stateOne->setTextureAttributeAndModes(0,panelTexture,osg::StateAttribute::ON); 
	panel->setStateSet(stateOne);

	// root node
	panelMatrixTransform = new osg::MatrixTransform;
	panelMatrixTransform->setName("MatrixTransform");
	panelMatrixTransform->setMatrix(osg::Matrix::scale(150, 100, 100));
	panelMatrixTransform->addChild(panel);

	return panelMatrixTransform;
}