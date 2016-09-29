#ifndef STREETVIEW_INDEX_H
#define STREETVIEW_INDEX_H

#include <string>
#include <vector>
#include <map>
#include <xercesc/dom/DOM.hpp>
namespace osg
{
	class Node;
};

class IndexParser;
class Picture;
class Station;
class Camera;
class StreetView;

class Index
{
public:
	Index(xercesc::DOMNode *indexNode, IndexParser *indexParser, StreetView *streetView);
	~Index();
	std::string &getDirection(){return direction;};
	std::string &getVersion(){return version;};
	std::string getAbsolutePicturePath();
	std::string &getPicturePath(){return picturePath;};
	std::string &getRoadName(){return roadName;};
	bool parsePictureIndex();
	void sortPicturesPerStation();
	std::vector<Camera *> cameraList;
	std::vector<Picture *> pictureList;
	std::vector<Station *> stationList;
	osg::Node *getNearestStationNode(double x, double y, double z);

private:
	int vonNetzKnoten;
	int nachNetzKnoten;
	std::string version;
	std::string direction;
	std::string picturePath;
	std::string roadName;
	IndexParser *indexParser;
	StreetView *streetView;
	// std::map<std::pair<double, double>, Station *> stations;
	std::vector<std::string> cameraSymbols;
	bool buildNewCamera(std::string currentCameraSymbol_);
};

#endif

