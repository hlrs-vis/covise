#include "Station.h"
#include "Picture.h"


Station::Station(Picture *picture_)
{
	picture = picture_;
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
	stationMatrixTransform->setName("set name for station"); // set name for station
	stationMatrixTransform->addChild(stationPanels);
	// stationMatrixTransform->setMatrix(); // add station's position

	return stationMatrixTransform;
}