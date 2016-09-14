#include "Camera.h"

#include <iostream>


Camera::Camera(xercesc::DOMElement *cameraElement_, const char cameraSymbol_)
{
	cameraSymbol = cameraSymbol_;
	cameraName = xercesc::XMLString::transcode(cameraElement_->getAttribute(xercesc::XMLString::transcode("name"))); // cameraElement_ = element "Region"

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
			const char *pitch = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pitch")));
			sscanf(pitch, "%lf", &d);
			rotation.push_back(d);
			const char *roll = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("roll")));
			sscanf(roll, "%lf", &d);
			rotation.push_back(d);
			const char *azimuth = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("azimut")));
			sscanf(azimuth, "%lf", &d);
			rotation.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Camera"), cameraElement->getTagName()))
		{
			const char *imageWidth = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("imagewidth")));
			sscanf(imageWidth, "%lf", &d);
			cameraImage.push_back(d);
			const char *imageHeight = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("imageheight")));
			sscanf(imageHeight, "%lf", &d);
			cameraImage.push_back(d);
			const char *pixelSizeX = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pixelsizex")));
			sscanf(pixelSizeX, "%lf", &d);
			cameraImage.push_back(d);		
			const char *pixelSizeY = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pixelsizey")));
			sscanf(pixelSizeY, "%lf", &d);
			cameraImage.push_back(d);
			const char *ck = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("ck")));
			sscanf(ck, "%lf", &d);
			cameraImage.push_back(d);
			const char *cx = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("cx")));
			sscanf(cx, "%lf", &d);
			cameraImage.push_back(d);
			const char *cy = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("cy")));
			sscanf(cy, "%lf", &d);
			cameraImage.push_back(d);
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
