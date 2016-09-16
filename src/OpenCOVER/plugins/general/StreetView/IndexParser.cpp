#include "IndexParser.h"
#include "Station.h"

#include <string>
#include <algorithm>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>

IndexParser::IndexParser()
{
}

IndexParser::~IndexParser(void)
{
	for(std::vector<Index *>::iterator it=indexList.begin();it != indexList.end();it++)
	{
		delete *it;
	}
}


bool IndexParser::parseIndex(std::string indexPath_)
{
	indexPath = indexPath_;
	//std::replace(indexPath.begin(), indexPath.end(), '\\', '/');
	std::string indexFile = indexPath+"/index.xml";

	xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
	parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

	try
	{
		parser->parse(indexFile.c_str());
	}
	catch (...)
	{
		fprintf(stderr, "error parsing xml file\n");
		return false;
	}

	xercesc::DOMDocument *xmlDoc = parser->getDocument();
	xercesc::DOMElement *rootElement = NULL;
	if (xmlDoc)
	{
		rootElement = xmlDoc->getDocumentElement();
	}
	if (rootElement)
	{
		xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
		for (int i = 0; i < nodeList->getLength(); ++i)
		{
			if (xercesc::DOMNode::ELEMENT_NODE == nodeList->item(i)->getNodeType())
			{
				xercesc::DOMElement *indexElement = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
				indexList.push_back(new Index(indexElement,this));
			}
		}
	}
	delete parser;
	return true;
}


void IndexParser::removeDuplicateEntries()
{
	std::vector<Index *>::iterator it = indexList.begin();
	if (it != indexList.end())
	{
		std::string stringToCompare = ((*it)->getPicturePath()+(*it)->getDirection());
		it++;
		while (it != indexList.end())
		{
			if (stringToCompare == ((*it)->getPicturePath()+(*it)->getDirection()))
			{
				it = indexList.erase(it);
			}
			else
			{
				stringToCompare = ((*it)->getPicturePath()+(*it)->getDirection());
				it++;
			}
		}
	}
}

/*/
// remove duplicate entries in cameraList
void IndexParser::removeDuplicateEntriesInCameras()
{
	std::vector<Camera *>::iterator it = cameraList.begin();
	if (it != cameraList.end())
	{
		std::string currentCamera = (*it)->getCameraName();
		it++;
		while (it != cameraList.end())
		{
			if (currentCamera == (*it)->getCameraName())
			{
				it = cameraList.erase(it);
			}
			else
			{
				currentCamera = (*it)->getCameraName();
				it++;
			}
		}
	}
}
/*/



void IndexParser::parsePictureIndices()
{
	std::vector<Index *>::iterator it = indexList.begin(); 
	if (it != indexList.end()) // debugging
	{
		(*it)->parsePictureIndex();
	}
	/*/
	for (std::vector<Index *>::iterator it = indexList.begin(); it != indexList.end(); it++)
	{
	(*it)->parsePictureIndex();
	}
	/*/
}


void IndexParser::sortIndicesPerStation()
{
	std::vector<Index *>::iterator it = indexList.begin(); 
	if (it != indexList.end()) // debugging
	{
		(*it)->sortPicturesPerStation();
	}
	/*/
	for (std::vector<Index *>::iterator it = indexList.begin(); it != indexList.end(); it++)
	{
		(*it)->sortPicturesPerStation();
	}
	/*/
}