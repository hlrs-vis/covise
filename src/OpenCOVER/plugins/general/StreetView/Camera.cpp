#include "Camera.h"

#include <iostream>


Camera::Camera(xercesc::DOMElement *cameraElement_, std::string cameraSymbol_)
{
	rotationPitch = 0.0;
	rotationRoll = 0.0;
	rotationAzimuth = 0.0;
	imageWidth = 0;
	imageHeight = 0;
	pixelSizeX = 0.0;
	pixelSizeY = 0.0;

	cameraSymbol = cameraSymbol_;
	cameraName = xercesc::XMLString::transcode(cameraElement_->getAttribute(xercesc::XMLString::transcode("name")));

	xercesc::DOMNodeList *cameraNodeList = cameraElement_->getChildNodes();
	for (int j = 0; j < cameraNodeList->getLength(); ++j)
	{
		double d = 0.0;
		xercesc::DOMElement *cameraElement = dynamic_cast<xercesc::DOMElement *>(cameraNodeList->item(j));
		if (!cameraElement)
			continue;
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("ProjectionCenter"), cameraElement->getTagName()))
		{
			const char *x = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("x")));
			sscanf(x, "%lf", &d);
			projectionCenter.push_back(d);
			const char *y = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("y")));
			sscanf(y, "%lf", &d);
			projectionCenter.push_back(d);
			const char *z = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("z")));
			sscanf(z, "%lf", &d);
			projectionCenter.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Rotation"), cameraElement->getTagName()))
		{
			const char *p = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pitch")));
			sscanf(p, "%lf", &rotationPitch);
			const char *r = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("roll")));
			sscanf(r, "%lf", &rotationRoll);
			const char *a = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("azimut")));
			sscanf(a, "%lf", &rotationAzimuth);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Camera"), cameraElement->getTagName()))
		{
			const char *iw = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("imagewidth")));
			sscanf(iw, "%d", &imageWidth);
			const char *ih = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("imageheight")));
			sscanf(ih, "%d", &imageHeight);
			const char *px = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pixelsizex")));
			sscanf(px, "%lf", &pixelSizeX);	
			const char *py = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pixelsizey")));
			sscanf(py, "%lf", &pixelSizeY);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Distortion"), cameraElement->getTagName()))
		{
			const char *k1 = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("k1")));
			sscanf(k1, "%lf", &d);
			distortion.push_back(d);
			const char *k2 = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("k2")));
			sscanf(k2, "%lf", &d);
			distortion.push_back(d);
			const char *p1 = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("p1")));
			sscanf(p1, "%lf", &d);
			distortion.push_back(d);		
			const char *p2 = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("p2")));
			sscanf(p2, "%lf", &d);
			distortion.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("SearchArea"), cameraElement->getTagName()))
		{
			const char *dx = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("dx")));
			sscanf(dx, "%lf", &d);
			distortion.push_back(d);
			const char *dy = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("dy")));
			sscanf(dy, "%lf", &d);
			distortion.push_back(d);
		}
	}
}

Camera::~Camera()
{
}
