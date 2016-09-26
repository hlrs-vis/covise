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

class Index
{
public:
	Index(xercesc::DOMNode *indexNode, IndexParser *indexParser);
	~Index();
	std::string &getDirection(){return direction;};
	std::string &getVersion(){return version;};
	std::string getAbsolutePicturePath();
	std::string &getPicturePath(){return picturePath;};
	bool parsePictureIndex();
	void sortPicturesPerStation();
	osg::Node *getStationNode(int stationNumber_);
	std::vector<Camera *> cameraList;
	std::vector<Picture *> pictureList;

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
	bool buildNewCamera(std::string currentCameraSymbol_);
};

#endif

