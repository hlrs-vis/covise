#include "Station.h"
#include "Picture.h"


Station::Station(Picture *picture_)
{
	picture = picture_;
	stationNumber = picture->getStation();
	stationLatitude = picture->getLatitude();
	stationLongitude = picture->getLongitude();
	stationPictures.push_back(picture);
}


Station::~Station()
{
}


osg::Node *Station::getStationPanels()
{
	stationPanels = new osg::Group;
	stationPanels->setName("Hier kommt der Stationsname rein"); // set name for station
	for (std::vector<Picture *>::iterator it = stationPictures.begin(); it != stationPictures.end(); it++)
	{
		stationPanels->addChild((*it)->getPanelNode());
	}
	osg::MatrixTransform *stationMatrixTransform = new osg::MatrixTransform;
	stationMatrixTransform->addChild(stationPanels);
	// stationMatrixTransform->setMatrix(); // add station's position

	return stationMatrixTransform;
}