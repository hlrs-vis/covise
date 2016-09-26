#ifndef STREET_VIEW_STATION_H
#define STREET_VIEW_STATION_H

#include <vector>
#include <osg/MatrixTransform>

class Picture;
class Index;

class Station
{
public:
	Station(Picture *picture_);
	~Station();
	osg::Node *getStationPanels();
	double &getStationLatitude(){return stationLatitude;};
	std::vector<Picture *> stationPictures;

private:
	int stationNumber;
	double stationLatitude;
	double stationLongitude;
	double stationAltitude;
	double stationHeading;
	Picture *picture;
	osg::ref_ptr<osg::MatrixTransform> stationMatrixTransform;
};

#endif