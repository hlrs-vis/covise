#ifndef INDEX_PARSER_H
#define INDEX_PARSER_H

#include "Index.h"
#include <vector>

class IndexParser
{
public:
	IndexParser(void);
	~IndexParser(void);
	/// parse index.xml in directory indexPath
	bool parseIndex(std::string indexPath);
	std::string &getIndexPath(){return indexPath;};
	std::vector<Index *> indexList;
	void parsePictureIndices();
	void removeDuplicateEntries();
private:
	std::string indexPath;
};
#endif


