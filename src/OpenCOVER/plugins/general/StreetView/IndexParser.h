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
	void parsePictureIndices();
	void parseCameras();
	void removeDuplicateEntries();
	void sortIndicesPerStation();
	std::vector<Index *> indexList;

private:
	std::string indexPath;
	StreetView *streetView;
};
#endif


