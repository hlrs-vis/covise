#ifndef STREETVIEW_CAMERA_H
#define STREETVIEW_CAMERA_H

#include <string>
#include <vector>

#include <xercesc/dom/DOM.hpp>

class Picture;
class Index;

class Camera
{
public:
	Camera(xercesc::DOMElement *cameraElement_, const char cameraSymbol_);
	~Camera();
	std::string &getCameraName(){return cameraName;};
	std::string &getCameraSymbol(){return cameraSymbol;};

private:
	std::string cameraName;
	std::string cameraSymbol;
	std::vector<double> projectionCenter; // x, y, z
	std::vector<double> rotation; // pitch, roll, azimuth
	std::vector<double> cameraImage; // width, height, pixelsizex, pixelsizey, ck, cx, cy
	std::vector<double> distortion; // k1, k2, p1, p2
	std::vector<double> searchArea; // dx, dy
};

#endif

