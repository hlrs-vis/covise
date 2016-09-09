#ifndef STREETVIEW_INDEX_H
#define STREETVIEW_INDEX_H

#include <string>
#include <xercesc/dom/DOM.hpp>
#include <vector>
class IndexParser;
class Picture;

class Index
{
public:
	Index(xercesc::DOMNode *indexNode,IndexParser *indexParser);
	~Index();
	std::string &getRichtung(){return richtung;};
	std::string &getVersion(){return version;};
	std::string getAbsolutePicturePath();
	std::string &getPicturePath(){return picturePath;};
	std::vector<Picture *> pictureList;
	bool parsePictureIndex();

private:
	int vonNetzKnoten;
	int nachNetzKnoten;
	std::string version;
	std::string richtung;
	std::string picturePath;
	std::string roadName;
	IndexParser *indexParser;
};

#endif

