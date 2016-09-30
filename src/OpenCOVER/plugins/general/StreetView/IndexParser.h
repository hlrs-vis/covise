#ifndef INDEX_PARSER_H
#define INDEX_PARSER_H

#include "Index.h"
#include "StreetView.h"

class IndexParser
{
public:
	IndexParser(StreetView *streetview);
	~IndexParser();
	bool parseIndex(std::string indexPath); 	// parse index.xml in directory indexPath
	std::string &getIndexPath(){return indexPath;};
	void parsePicturesPerStreet(std::string roadName_);
	void parseCameras();
	void removeDuplicateEntries();
	void sortStreetPicturesPerStation(); // could be done in Index->parsePictureIndex
	std::vector<Index *> indexList;
	std::vector<Index *> streetList;
	osg::Node *getNearestStationNode(double x, double y, double z); // input: viewer's position plus offset
	// void parsePictureIndices();
	// void sortIndicesPerStation();

private:
	std::string indexPath;
	StreetView *streetView;
};
#endif


