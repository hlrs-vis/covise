#include "Station.h"
#include "Picture.h"

#include <string>


Station::Station(Picture *picture_, Index *index_)
{
	picture = picture_;
	index = index_;
	stationNumber = picture->getStation();
	stationLatitude = picture->getLatitude();
	stationLongitude = picture->getLongitude();
	stationAltitude = picture->getAltitude();
	stationHeading = picture->getHeading();
	stationPictures.push_back(picture);
}


Station::~Station()
{
}


osg::Node *Station::getStationPanels()
{
	osg::Group *stationPanels = new osg::Group();
	for (std::vector<Picture *>::iterator it = stationPictures.begin(); it != stationPictures.end(); it++)
	{
		stationPanels->addChild((*it)->getPanelNode());
	}
	stationMatrixTransform = new osg::MatrixTransform;
	stationMatrixTransform->setName("Latitude_" + std::to_string(stationLatitude));
	stationMatrixTransform->addChild(stationPanels);
	
	/*/ check
	osg::Matrix rotationMatrix;
	rotationMatrix.makeRotate((stationHeading * M_PI / 180), 0, 0, 1);
	stationMatrixTransform->setMatrix(rotationMatrix);
	/*/

	return stationMatrixTransform;
}