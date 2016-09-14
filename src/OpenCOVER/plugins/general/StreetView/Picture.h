#ifndef STREET_VIEW_PICTURE_H
#define STREET_VIEW_PICTURE_H

#include <string>
#include <xercesc/dom/DOM.hpp>
#include <osg/MatrixTransform>

class Camera;
class Index;

class Picture
{
public:
	Picture(xercesc::DOMNode *pictureNode, Index *index);
	~Picture();
	osg::Node *getPanelNode();
	std::string getCameraSymbol(){return cameraSymbol;};
	int &getStation(){return station;};
	double &getLatitude(){return latitude;};
	double &getLongitude(){return longitude;};

private:
	Camera *camera;
	Index *index;
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

