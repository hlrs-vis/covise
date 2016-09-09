#include "IndexParser.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>

IndexParser::IndexParser(void)
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
		    xercesc::DOMElement *indexElement = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
			if(indexElement)
			{
				indexList.push_back(new Index(indexElement,this));
			}
		}
	}
	delete parser;
	return true;
}


void IndexParser::parsePictureIndices()
{
   for(std::vector<Index *>::iterator it = indexList.begin(); it != indexList.end();it++)
   {
	   (*it)->parsePictureIndex();
   }
}
void IndexParser::removeDuplicateEntries()
{
}
