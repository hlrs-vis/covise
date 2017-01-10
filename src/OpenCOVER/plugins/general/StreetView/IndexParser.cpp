#include "IndexParser.h"
#include "Station.h"

#include <string>
#include <algorithm>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>

IndexParser::IndexParser(StreetView *streetview_)
{
	streetView = streetview_;
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
				indexList.push_back(new Index(indexElement, this, streetView));
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

void IndexParser::parsePicturesPerStreet(std::string roadName_)
{
	std::vector<Index *>::iterator it = indexList.begin(); 
	if (it != indexList.end())
	{
		while (it != indexList.end() && (*it)->getRoadName() != roadName_)
		{
			it++;
		}
		while (it != indexList.end() && (*it)->getRoadName() == roadName_)
		{
			streetList.push_back(*it);
			streetList.back()->parsePictureIndex();
			it++;
		}
	}
}

void IndexParser::sortStreetPicturesPerStation()
{
	for (std::vector<Index *>::iterator it = streetList.begin(); it != streetList.end(); it++)
	{
		(*it)->sortPicturesPerStation();
	}
}

osg::Node *IndexParser::getNearestStationNode(double x, double y, double z)
{
	Station *nearestStationPerIndex = NULL;
	double distance = -1.0;
	double tmpDistance = -1.0;
	for (std::vector<Index *>::iterator it = streetList.begin(); it != streetList.end(); it++)
	{
		if (distance == -1.0 || (tmpDistance != -1.0 && tmpDistance < distance))
		{
			nearestStationPerIndex = (*it)->getNearestStationPerIndex(x, y, z);
			tmpDistance = (*it)->getMinimumDistance();
			distance = tmpDistance;	
		}
	}
	if (nearestStationPerIndex)
	{
		return nearestStationPerIndex->getStationPanels();
	}
	return NULL;
}

/*
void IndexParser::parsePictureIndices()
{
	for (std::vector<Index *>::iterator it = indexList.begin(); it != indexList.end(); it++)
	{
		(*it)->parsePictureIndex();
	}

}

void IndexParser::sortIndicesPerStation()
{
	for (std::vector<Index *>::iterator it = indexList.begin(); it != indexList.end(); it++)
	{
		(*it)->sortPicturesPerStation();
	}
}
*/