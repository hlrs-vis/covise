#include "Camera.h"

#include <iostream>


Camera::Camera(xercesc::DOMElement *cameraElement_, std::string cameraSymbol_)
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
			const char *tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("x")));
			sscanf(tmp, "%lf", &d);
			projectionCenter.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("y")));
			sscanf(tmp, "%lf", &d);
			projectionCenter.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("z")));
			sscanf(tmp, "%lf", &d);
			projectionCenter.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Rotation"), cameraElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pitch")));
			sscanf(tmp, "%lf", &d);
			rotation.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("roll")));
			sscanf(tmp, "%lf", &d);
			rotation.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("azimut")));
			sscanf(tmp, "%lf", &d);
			rotation.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Camera"), cameraElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("imagewidth")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("imageheight")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pixelsizex")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);		
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("pixelsizey")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("ck")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("cx")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("cy")));
			sscanf(tmp, "%lf", &d);
			cameraImage.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Distortion"), cameraElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("k1")));
			sscanf(tmp, "%lf", &d);
			distortion.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("k2")));
			sscanf(tmp, "%lf", &d);
			distortion.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("p1")));
			sscanf(tmp, "%lf", &d);
			distortion.push_back(d);		
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("p2")));
			sscanf(tmp, "%lf", &d);
			distortion.push_back(d);
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("SearchArea"), cameraElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("dx")));
			sscanf(tmp, "%lf", &d);
			distortion.push_back(d);
			tmp = xercesc::XMLString::transcode(cameraElement->getAttribute(xercesc::XMLString::transcode("dy")));
			sscanf(tmp, "%lf", &d);
			distortion.push_back(d);
		}
	}
}

Camera::~Camera()
{
}
