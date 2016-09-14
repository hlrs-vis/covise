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


osg::Node *Station::getStationPanelGroup()
{
	stationPanelGroup = new osg::Group;
	stationPanelGroup->setName("StationPanelGroup");
	for (std::vector<Picture *>::iterator it = stationPictures.begin(); it != stationPictures.end(); it++)
	{
		stationPanelGroup->addChild((*it)->getPanelNode());
	}
	return stationPanelGroup;
}