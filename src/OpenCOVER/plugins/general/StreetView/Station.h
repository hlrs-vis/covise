#ifndef STREET_VIEW_STATION_H
#define STREET_VIEW_STATION_H

#include <vector>
#include <osg/MatrixTransform>

class Picture;
class Index;

class Station
{
public:
	Station(Picture *picture, Index *index);
	~Station();
	osg::Node *getStationPanels();
	int &getStationNumber(){return stationNumber;};
	double &getStationLongitude(){return stationLongitude;};
	double &getStationLatitude(){return stationLatitude;};
	double &getStationAltitude(){return stationAltitude;};
	std::vector<Picture *> stationPictures;

private:
	int stationNumber;
	double stationLongitude;
	double stationLatitude;
	double stationAltitude;
	double stationHeading;
	Picture *picture;
	Index *index;
	osg::ref_ptr<osg::MatrixTransform> stationMatrixTransform;
};

#endif