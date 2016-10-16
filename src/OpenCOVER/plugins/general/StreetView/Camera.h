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
	Camera(xercesc::DOMElement *cameraElement_, std::string cameraSymbol_);
	~Camera();
	std::string &getCameraName(){return cameraName;};
	std::string &getCameraSymbol(){return cameraSymbol;};
	double &getRotationPitch(){return rotationPitch;};
	double &getRotationRoll(){return rotationRoll;};
	double &getRotationAzimuth(){return rotationAzimuth;};
	int &getImageWidth(){return imageWidth;};
	int &getImageHeight(){return imageHeight;};
	double &getPixelSizeX(){return pixelSizeX;};
	double &getPixelSizeY(){return pixelSizeY;};

private:
	std::string cameraName;
	std::string cameraSymbol;
	std::vector<double> projectionCenter; // x, y, z
	std::vector<double> distortion; // k1, k2, p1, p2
	std::vector<double> searchArea; // dx, dy
	double rotationPitch;
	double rotationRoll;
	double rotationAzimuth;
	int imageWidth;
	int imageHeight;
	double pixelSizeX;
	double pixelSizeY;
};

#endif

