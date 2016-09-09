#include "Index.h"
#include "IndexParser.h"
#include "Picture.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <iostream>

Index::Index(xercesc::DOMNode *indexNode, IndexParser *indexParser_)
{
	vonNetzKnoten = 0;
	nachNetzKnoten = 0;
	version = '0';
	indexParser=indexParser_;
	
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
			richtung = xercesc::XMLString::transcode(indexElement->getTextContent());
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
	for(std::vector<Picture *>::iterator it=pictureList.begin();it != pictureList.end();it++)
	{
		delete *it;
	}
}


std::string Index::getAbsolutePicturePath()
{
	return (indexParser->getIndexPath()+"/"+picturePath);
}

bool Index::parsePictureIndex()
{
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

	try
    {
        parser->parse((getAbsolutePicturePath()+"/"+"PIC_"+richtung+version+".xml").c_str());
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
		    xercesc::DOMElement *pictureElement = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
			if(pictureElement!=NULL)
			{
				pictureList.push_back(new Picture(pictureElement,this));
			}
			// setPicturePath
		}
	}
	delete parser;
	return true;
}
