#include <generateParser.h>
#include <string>
#include <iostream>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>

void parseElement(xercesc::DOMElement *elem)
{

	std::string name;
	name = xercesc::XMLString::transcode(elem->getNodeName());
	std::cerr << "" << name;

	xercesc::DOMNodeList *elementList = elem->getChildNodes();

	for (unsigned int child = 0; child < elementList->getLength(); ++child)
	{

		xercesc::DOMElement *element = dynamic_cast<xercesc::DOMElement *>(elementList->item(child));
		if (element)
		{
			parseElement(element);
		}
	}
}

void parseSchema(xercesc::DOMElement *elem)
{

	std::string name;
	name = xercesc::XMLString::transcode(elem->getNodeName());
	std::cerr << "" << name;

	xercesc::DOMNodeList *elementList = elem->getChildNodes();

	for (unsigned int child = 0; child < elementList->getLength(); ++child)
	{

		xercesc::DOMElement *element = dynamic_cast<xercesc::DOMElement *>(elementList->item(child));
		if (element)
		{

			std::string name;
			name = xercesc::XMLString::transcode(element->getNodeName());
			if (name == "xsd:element")
			{
				parseElement(element);
			}
		}
	}
}

int main(int argc, char **argv)
{
	xercesc::XercesDOMParser *parser;
	//in order to work with the Xerces-C++ parser, the XML subsystem must be initialized first
	//every call of XMLPlatformUtils::Initialize() must have a matching call of XMLPlatformUtils::Terminate() (see destructor)
	try
	{
		xercesc::XMLPlatformUtils::Initialize();
	}
	catch (const xercesc::XMLException &toCatch)
	{
		char *message = xercesc::XMLString::transcode(toCatch.getMessage());
		std::cerr << "Error during xerces initialization! :\n" << message << std::endl;
		xercesc::XMLString::release(&message);
	}

	//parser and error handler have to be initialized _after_ xercesc::XMLPlatformUtils::Initialize()
	//can't be done in member initializer list
	parser = new xercesc::XercesDOMParser();

	//generic settings for parser
	//
//	parser->setErrorHandler(parserErrorHandler);
//	parser->setExitOnFirstFatalError(true);
	//namespaces needed for XInclude and validation
	//settings for validation

	parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);
	parser->setDoNamespaces(true);
	//parser->setUserEntityHandler(fEntityHandler);
	//parser->setUserErrorReporter(fErrorReporter);

	parser->setDoXInclude(true);

	//parse the file
	try
	{
		parser->parse(argv[1]);
	}
	catch (...)
	{
		std::cerr << "\nErrors during parse of the document '" << argv[1] << "'.\n" << std::endl;

		return 1;
	}
	xercesc::DOMNodeList *elementList = parser->getDocument()->getChildNodes();

	for (unsigned int child = 0; child < elementList->getLength(); ++child)
	{

		xercesc::DOMElement *element = dynamic_cast<xercesc::DOMElement *>(elementList->item(child));
		if (element)
		{

			char *xname = xercesc::XMLString::transcode(element->getNodeName());
            std::string name(xname);
            xercesc::XMLString::release(&xname);
			if (name == "xsd:schema")
			{
				parseSchema(element);
			}
		}
	}

    return 0;
}
