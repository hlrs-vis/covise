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

private:
	Camera *camera;
	std::string fileName;
	double latitude;
	double longitude;
	double altitude;
	double heading;
	std::string PicturePath;
	std::string absPicturePath;
	Index *index;
	
	osg::ref_ptr<osg::MatrixTransform> viereckMatrixTransform;
};

#endif

