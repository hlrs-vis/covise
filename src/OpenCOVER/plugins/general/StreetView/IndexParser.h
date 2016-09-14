#ifndef INDEX_PARSER_H
#define INDEX_PARSER_H

#include "Index.h"
#include "StreetView.h"

class IndexParser
{
public:
	IndexParser();
	~IndexParser();
	bool parseIndex(std::string indexPath); 	// parse index.xml in directory indexPath
	std::string &getIndexPath(){return indexPath;};
	std::vector<Index *> indexList;
	void parsePictureIndices();
	void parseCameras();
	void removeDuplicateEntries();
	void IndexParser::sortIndicesPerStation();

private:
	std::string indexPath;
	StreetView *streetView;
};
#endif


