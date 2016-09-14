#ifndef STREET_VIEW_STATION_H
#define STREET_VIEW_STATION_H

#include <vector>
#include <osg/MatrixTransform>

class Picture;
class Index;

class Station
{
public:
	Station(Index *index_);
	Station(Picture *picture_);
	~Station();
	osg::Node *getStationPanelGroup();
	std::vector<Picture *> stationPictures;

private:
	int stationNumber;
	double stationLatitude;
	double stationLongitude;
	Picture *picture;
	osg::ref_ptr<osg::Group> stationPanelGroup;
};

#endif