#include "Index.h"
#include "IndexParser.h"
#include "Picture.h"
#include "Station.h"
#include "Camera.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

namespace fs = ::boost::filesystem;


Index::Index(xercesc::DOMNode *indexNode, IndexParser *indexParser_)
{
	vonNetzKnoten = 0;
	nachNetzKnoten = 0;
	version = '0';
	indexParser = indexParser_;

	// parse a single index node
	xercesc::DOMNodeList *indexNodeList = indexNode->getChildNodes();
	for (int j = 0; j < indexNodeList->getLength(); ++j)
	{
		xercesc::DOMElement *indexElement = dynamic_cast<xercesc::DOMElement *>(indexNodeList->item(j));
		if (!indexElement)
			continue;
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("VNK"), indexElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(indexElement->getTextContent());
			sscanf(tmp, "%d", &vonNetzKnoten);
		}	
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("NNK"), indexElement->getTagName()))
		{
			const char *tmp = xercesc::XMLString::transcode(indexElement->getTextContent());
			sscanf(tmp, "%d", &nachNetzKnoten);
		}	
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("VERSION"), indexElement->getTagName()))
		{
			version = xercesc::XMLString::transcode(indexElement->getTextContent());
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("RICHTUNG"), indexElement->getTagName()))
		{
			direction = xercesc::XMLString::transcode(indexElement->getTextContent());
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("PICPATH"), indexElement->getTagName()))
		{
			picturePath = xercesc::XMLString::transcode(indexElement->getTextContent());
		}
		if(xercesc::XMLString::equals(xercesc::XMLString::transcode("Roadname"), indexElement->getTagName()))
		{
			roadName = xercesc::XMLString::transcode(indexElement->getTextContent());
		}
	}
}


Index::~Index()
{
	for (std::vector<Picture *>::iterator it = pictureList.begin(); it != pictureList.end(); it++)
	{
		delete *it;
	}

	for (std::vector<Camera *>::iterator it = cameraList.begin(); it != cameraList.end(); it++)
	{
		delete *it;
	}

	// map stations
}


std::string Index::getAbsolutePicturePath()
{
	return (indexParser->getIndexPath()+"/"+picturePath);
}


bool Index::parsePictureIndex()
{
	xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
	parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

	// parse pictures in current directory
	try
	{
		parser->parse((getAbsolutePicturePath()+"/"+"PIC_"+direction+version+".xml").c_str());
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
			if(xercesc::DOMNode::ELEMENT_NODE == nodeList->item(i)->getNodeType())
			{
				xercesc::DOMElement *pictureElement = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
				Picture *currentPicture = new Picture(pictureElement,this);
				pictureList.push_back(currentPicture);
				std::string currentCameraSymbol = currentPicture->getCameraSymbol();

				// search in cameraList for camera. build one, if not found
				std::vector<Camera *>::iterator it = cameraList.begin();
				if (it != cameraList.end())
				{
					while ((it != cameraList.end()) && ((*it)->getCameraSymbol() != currentCameraSymbol))
					{
						it++;
					}
					if (it == cameraList.end())
					{
						buildNewCamera(currentCameraSymbol);
					}
					else
					{
						pictureList.back()->setCamera(*it);
					}
				}
				else
				{
					buildNewCamera(currentCameraSymbol);
				}
			}
		}
	}
	delete parser;
	return true;
}

bool Index::buildNewCamera(std::string currentCameraSymbol_)
{
	std::string currentCameraSymbol = currentCameraSymbol_;

	xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
	parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

	try
	{
		parser->parse((getAbsolutePicturePath()+"/"+direction+currentCameraSymbol+version+".cam").c_str());
	}
	catch (...)
	{
		fprintf(stderr, "error parsing cam file\n");
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
		rootElement = rootElement->getFirstElementChild();
		if (rootElement)
		{
			cameraList.push_back(new Camera(rootElement, currentCameraSymbol));
			pictureList.back()->setCamera(cameraList.back());
		}
	}
	delete parser;
	return true;
}


void Index::sortPicturesPerStation()
{
	for (std::vector<Picture *>::iterator it = pictureList.begin(); it != pictureList.end(); it++)
	{
		int stationNumber = (*it)->getStation();
		if (stations.count(stationNumber) == 0)
		{
			stations.insert(std::pair<int, Station *> (stationNumber, new Station(*it)));
		}
		else
		{
			stations.find(stationNumber)->second->stationPictures.push_back(*it);
		}
	}
}

// besser: new myStation, insert und methode add, erst nach Bedingung push_back


osg::Node *Index::getStationNode(int stationNumber_)
{
	if (stations.count(stationNumber_) != 0)
	{
		return stations.find(stationNumber_)->second->getStationPanels();
	}
	else
	{
		return NULL;
	}
}
