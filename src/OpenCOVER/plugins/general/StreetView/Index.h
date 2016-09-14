#ifndef STREETVIEW_INDEX_H
#define STREETVIEW_INDEX_H

#include <string>
#include <vector>
#include <map>
#include <xercesc/dom/DOM.hpp>

class IndexParser;
class Picture;
class Station;
class Camera;

class Index
{
public:
	Index(xercesc::DOMNode *indexNode, IndexParser *indexParser);
	~Index();
	std::string &getDirection(){return direction;};
	std::string &getVersion(){return version;};
	std::string getAbsolutePicturePath();
	std::string &getPicturePath(){return picturePath;};
	std::vector<Picture *> pictureList;
	bool parsePictureIndex();
	bool parseCamerasPerIndex();
	void sortPicturesPerStation();
	std::vector<Camera *> cameraList;
	Station *getStation(int stationNumber_);

private:
	int vonNetzKnoten;
	int nachNetzKnoten;
	std::string version;
	std::string direction;
	std::string picturePath;
	std::string roadName;
	IndexParser *indexParser;
	std::map<int, Station *> stations;
	std::vector<std::string> cameraSymbols;
};

#endif

