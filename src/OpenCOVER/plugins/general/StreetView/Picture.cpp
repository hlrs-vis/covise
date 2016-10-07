#include "Picture.h"
#include "Index.h"
#include "Camera.h"
#include "StreetView.h"
#include "Position.h"

#include <osgDB/ReadFile>

#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Texture2D>

#include <iostream>
#include <cmath>

Picture::Picture(xercesc::DOMNode *pictureNode, Index *index_, StreetView *streetview_)
{
	station = 0;
	latitude = 0.0;
	longitude = 0.0;
	altitude = 0.0;
	heading = 0.0;
	index = index_;
	streetView = streetview_;
	camera = NULL;
	Position *picturePosition = new Position;
	
	xercesc::DOMNodeList *pictureNodeList = pictureNode->getChildNodes();
	for (int j = 0; j < pictureNodeList->getLength(); ++j)
	{
		xercesc::DOMElement *pictureElement = dynamic_cast<xercesc::DOMElement *>(pictureNodeList->item(j));
		if (!pictureElement)
			continue;
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Buchst"), pictureElement->getTagName()))
		{
			cameraSymbol = xercesc::XMLString::transcode(pictureElement->getTextContent());
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

	if (longitude != 0.0 && latitude != 0.0 && altitude != 0.0)
	{
		picturePosition->transformWGS84ToGauss(longitude, latitude, altitude);
	}
}


Picture::~Picture()
{
	delete camera;
	delete streetView;
	delete index;
}


osg::Node *Picture::getPanelNode()
{
	osg::Geode *panel = new osg::Geode();
	osg::Geometry *panelGeometry = new osg::Geometry();
	
	// a = 2*arctan(g/(2r)),  g = 2r*tan(a/2) // r = distance, a = FOV
	// a = 2*atan(d/f) // general FOV, d = picture diagonal, f = focal length
	// H = 2*atan(tan(V/2)*(w/h)) // horizontal FOV
	// V = 2*atan(tan(H/2)*(h/w)) // vertical FOV
	// H or V = 2*arctan (B/2 * 1/f´) (in rad)

	int focalLength = 50; // assumed
	double distance = 4; // in metres, assumed
	double aspectRatioWH = (1280.0*(camera->getPixelSizeX()))/(960.0*(camera->getPixelSizeY()));
	double aspectRatioHW = (960.0*(camera->getPixelSizeY()))/(1280.0*(camera->getPixelSizeX()));
	double alphaV = 7.3; // vertical FOV, based on focalLength
	double alphaH = 2.0*atan(tan(alphaV/2.0)*(aspectRatioWH)); // horizontal FOV
	double width = 2.0*distance*tan(alphaH/2.0);
	double height = 2.0*distance*tan(alphaV/2.0);

	osg::Vec3Array *panelVertices = new osg::Vec3Array;
	panelVertices->reserve(4);
	panelVertices->push_back(osg::Vec3(-width/2.0,  0, -height/2.0));
	panelVertices->push_back(osg::Vec3( width/2.0,  0, -height/2.0));
	panelVertices->push_back(osg::Vec3( width/2.0,  0,  height/2.0));
	panelVertices->push_back(osg::Vec3(-width/2.0,  0,  height/2.0));
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
	stateOne->setTextureAttributeAndModes(0, panelTexture, osg::StateAttribute::ON); 
	panel->setStateSet(stateOne);

	osg::Matrix rotationMatrix;
	rotationMatrix.makeRotate(
		camera->getRotationPitch(), osg::Vec3(1,0,0), 
		camera->getRotationRoll(), osg::Vec3(0,1,0), 
		camera->getRotationAzimuth(), osg::Vec3(0,0,1));

	osg::Matrix translationMatrix;
	translationMatrix.makeTranslate(distance, distance, 0);

	osg::Matrix rotationMatrix90X;
	rotationMatrix90X.makeRotate(1.570796, -1, 0, 0);

	osg::Matrix rotationMatrix90Y;
	rotationMatrix90Y.makeRotate(1.570796, 0, -1, 0);

	// root node
	panelMatrixTransform = new osg::MatrixTransform;
	panelMatrixTransform->setName(fileName);
	panelMatrixTransform->addChild(panel);
	panelMatrixTransform->setMatrix(rotationMatrix90X*rotationMatrix90Y*translationMatrix*rotationMatrix);

	return panelMatrixTransform;
}