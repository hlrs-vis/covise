#ifndef STREET_VIEW_PICTURE_H
#define STREET_VIEW_PICTURE_H

#include <string>
#include <xercesc/dom/DOM.hpp>
#include <osg/MatrixTransform>

class Camera;
class Index;
class StreetView;

class Picture
{
public:
	Picture(xercesc::DOMNode *pictureNode, Index *index, StreetView *streetView);
	~Picture();
	osg::Node *getPanelNode();
	std::string getCameraSymbol(){return cameraSymbol;};
	int &getStation(){return station;};
	double &getLatitude(){return latitude;};
	double &getLongitude(){return longitude;};
	double &getAltitude(){return altitude;};
	double &getHeading(){return heading;};
	Camera *getCamera(){return camera;};
	void setCamera(Camera *currentCamera_){camera = currentCamera_;};

private:
	Camera *camera;
	Index *index;
	StreetView *streetView;
	std::string cameraSymbol;
	int station;
	std::string fileName;
	double latitude;
	double longitude;
	double altitude;
	double heading;
	osg::ref_ptr<osg::MatrixTransform> panelMatrixTransform;
};

#endif

