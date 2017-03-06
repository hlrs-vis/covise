#include <generateParser.h>
#include <string>
#include <iostream>
#include <list>
#include <stack>
#include <vector>
#include <algorithm>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include "../oscNameMapping.h"

class oscClass;
class oscMember;
class oscEnum;
oscEnum *currentEnum = NULL;
std::stack<oscClass *> currentClassStack;
oscClass *currentClass = NULL;
oscMember *currentMember = NULL;
typedef std::pair<std::string, std::string> epair;
std::list<epair> enumRename;
std::string elementName;
std::string parentName;

std::string trim(const std::string& str)
{
	size_t first = str.find_first_not_of(' ');
	if (first == std::string::npos)
		return "";
	size_t last = str.find_last_not_of(' ');
	return str.substr(first, (last - first + 1));
}


std::list<oscClass *> classes;
std::list<oscEnum *> enums;

class oscMember
{
public:
	oscMember(std::string name);
	~oscMember();
	std::string name;
	std::string type;
	bool optional;
	int choice;
	bool array;
};

class oscClass
{
public:
	oscClass(std::string name);
	~oscClass();
	std::string name;
	std::string parentName;
	std::list<oscMember *> members;
	std::list<oscMember *> attributes;
	std::list<std::string> enumHeaders;
};

class oscEnum
{
public:
	oscEnum(std::string name);
	~oscEnum();
	std::string name;
	std::list<std::string> values;
	oscClass *ownerClass;
};

std::string makeTypeName(const std::string &n)
{
	std::string name=trim(n);
	if (name[0] == 'x'&&name[1] == 's'&&name[2] == 'd')
	{
	}
	else if (name[0] == 'E'&&name[1] == 'n'&&name[2] == 'u'&&name[3] == 'm')
	{
	}
	else if (name[0] == 'O'&&name[1] == 'S'&&name[2] == 'C')
	{
		name[0] = 'o';
		name[1] = 's';
		name[2] = 'c';
	}
	else
	{
		name = "osc" + name;
	}
	return name;
}
oscMember::oscMember(std::string n)
{
	name = n;
	std::string tn = OpenScenario::nameMapping::instance()->getClassName(n, parentName);
	type = makeTypeName(tn);
	choice = 0;
	optional = false;
	array = false;
	//std::cerr << "Member " << name << std::endl;
}

oscMember::~oscMember()
{
}


oscClass::oscClass(std::string n)
{
	name = makeTypeName(n);
	classes.push_back(this);

	//std::cerr << "Class " << name << std::endl;
}

oscClass::~oscClass()
{
}

oscEnum::oscEnum(std::string n)
{
	name = makeTypeName(n);
	enums.push_back(this);
	ownerClass = NULL;
	//std::cerr << "Enum " << name << std::endl;
}

oscEnum::~oscEnum()
{
}


void parseComplexType(xercesc::DOMElement *elem, int choice);
void parseSimpleType(xercesc::DOMElement *elem, int choice);
void parseGeneric(xercesc::DOMElement *elem, int choice);
void parseElement(xercesc::DOMElement *elem, int choice);
void parseAttribute(xercesc::DOMElement *elem, int choice);
void parseSchema(xercesc::DOMElement *elem);
void parseChoice(xercesc::DOMElement *elem);

void parseElement(xercesc::DOMElement *elem, int choice)
{
	xercesc::DOMAttr *attribute = elem->getAttributeNode(xercesc::XMLString::transcode("name"));
	if (attribute)
	{
		elementName = xercesc::XMLString::transcode(attribute->getValue());

		currentMember = new oscMember(elementName);
		if (currentClass)
			currentClass->members.push_back(currentMember);
	}
	else
	{
		std::cerr << "element without name attribute" << std::endl;
	}

	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("type"));
	if (attribute)
	{
		std::string tn = xercesc::XMLString::transcode(attribute->getValue());
		if (tn == "xsd:string")
		{
			currentMember->type = makeTypeName(std::string("StringElement"));
		}
		else
		{
			currentMember->type = makeTypeName(tn);
		}
	}
	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("maxOccurs"));
	if (attribute)
	{
		currentMember->array = true;
	}
	std::string minOccurs;
	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("minOccurs"));
	if (attribute)
	{
		minOccurs = xercesc::XMLString::transcode(attribute->getValue());
		if (minOccurs == "0")
		{
			currentMember->optional = true;
		}
	}

	currentMember->choice = choice;

	parseGeneric(elem, 0);
}

void parseChoice(xercesc::DOMElement *elem, int choice)
{

	parseGeneric(elem, choice);
}

void parseAttribute(xercesc::DOMElement *elem, int choice)
{
	xercesc::DOMAttr *attribute = elem->getAttributeNode(xercesc::XMLString::transcode("name"));
	if (attribute)
	{
		elementName = xercesc::XMLString::transcode(attribute->getValue());

		currentMember = new oscMember(elementName);

		if (currentClass)
			currentClass->attributes.push_back(currentMember);
	}
	else
	{
		std::cerr << "element without name attribute" << std::endl;
	}

	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("type"));
	if (attribute)
	{
		std::string tn = xercesc::XMLString::transcode(attribute->getValue());
		currentMember->type = makeTypeName(tn);
	}
	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("maxOccurs"));
	if (attribute)
	{
		currentMember->array = true;
	}
	std::string minOccurs;
	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("minOccurs"));
	if (attribute)
	{
		minOccurs = xercesc::XMLString::transcode(attribute->getValue());
		if (minOccurs == "0")
		{
			currentMember->optional = true;
		}
	}
	std::string use;
	attribute = elem->getAttributeNode(xercesc::XMLString::transcode("use"));
	if (attribute)
	{
		use = xercesc::XMLString::transcode(attribute->getValue());
		if (use == "optional")
		{
			currentMember->optional = true;
		}
		else
		{
			currentMember->optional = false;
		}
	}

	currentMember->choice = choice;

	parseGeneric(elem, 0);
}
void parseComplexType(xercesc::DOMElement *elem, int choice)
{
	std::string name;
	xercesc::DOMAttr *attribute = elem->getAttributeNode(xercesc::XMLString::transcode("name"));
	if (attribute)
	{
		name = xercesc::XMLString::transcode(attribute->getValue());
	}
	else
	{
		name = elementName;
	}
	name = OpenScenario::nameMapping::instance()->getClassName(name, parentName);
	currentClass = new oscClass(name);
	currentClass->parentName = parentName;
	currentClassStack.push(currentClass);
	parentName = parentName+"/"+name;
	parseGeneric(elem, choice);
}

void parseSimpleType(xercesc::DOMElement *elem, int choice)
{
	std::string name;
	xercesc::DOMAttr *attribute = elem->getAttributeNode(xercesc::XMLString::transcode("name"));
	if (attribute)
	{
		name = xercesc::XMLString::transcode(attribute->getValue());
	}
	else
	{
		name = elementName;
	}
	currentEnum = new oscEnum(name);
	parseGeneric(elem, choice);
}
void parseEnum(xercesc::DOMElement *elem)
{
	std::string value;
	xercesc::DOMAttr *attribute = elem->getAttributeNode(xercesc::XMLString::transcode("value"));
	if (attribute)
	{
		value = xercesc::XMLString::transcode(attribute->getValue());
		for (std::list<epair>::iterator it = enumRename.begin(); it != enumRename.end(); it++)
		{
			if (it->first == value)
			{
				value = it->second;
				break;
			}
		}
		currentEnum->values.push_back(value);
	}
	parseGeneric(elem, 0);
}


void parseGeneric(xercesc::DOMElement *elem, int choice)
{

	std::string name;
	name = xercesc::XMLString::transcode(elem->getNodeName());
	xercesc::DOMNodeList *elementList = elem->getChildNodes();
	std::string myParentName = parentName;

	for (unsigned int child = 0; child < elementList->getLength(); ++child)
	{
		int currentChoice = 0;
		xercesc::DOMElement *element = dynamic_cast<xercesc::DOMElement *>(elementList->item(child));
		if (element)
		{
			std::string name;
			name = xercesc::XMLString::transcode(element->getNodeName());

			parentName = myParentName;

			if (name == "xsd:element")
			{
				parseElement(element, choice);
			}
			else if (name == "xsd:attribute")
			{
				parseAttribute(element, choice);
			}
			else if (name == "xsd:complexType")
			{
				parseComplexType(element, choice);

				currentClassStack.pop();
				if (currentClassStack.size() > 0)
					currentClass = currentClassStack.top();
				else
					currentClass = NULL;
			}
			else if (name == "xsd:simpleType")
			{
				parseSimpleType(element, choice);
			}
			else if (name == "xsd:enumeration")
			{
				parseEnum(element);
			}
			else if (name == "xsd:choice")
			{
				currentChoice++;
				parseChoice(element, currentChoice);
			}
			else
			{
				parseGeneric(element,0);
			}
		}
	}
}

void parseSchema(xercesc::DOMElement *elem)
{

	std::string name;
	name = xercesc::XMLString::transcode(elem->getNodeName());
	std::cerr << "" << name;


	parseGeneric(elem, 0);
}

int main(int argc, char **argv)
{
	enumRename.push_back(std::pair<std::string, std::string>("greater-than", "greater_than"));
	enumRename.push_back(std::pair<std::string, std::string>("equal-to", "equal_to"));
	enumRename.push_back(std::pair<std::string, std::string>("less-than", "less_than"));
	enumRename.push_back(std::pair<std::string, std::string>("sky off", "sky_off"));
	enumRename.push_back(std::pair<std::string, std::string>("int", "int_t"));
	enumRename.push_back(std::pair<std::string, std::string>("double", "double_t"));
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

	oscClass * currentClass = new oscClass("None");
	currentClass = new oscClass("Polyline");
	currentClass = new oscClass("Delete");
	FILE *duplicates = fopen("duplicateClasses.txt", "w");
	for (int i = 1; i < argc; i++)
	{

		elementName="";
		parentName="";
		if (strstr(argv[i],"Catalog")!=NULL)
		{
			parentName = "/Catalog";
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
			parser->parse(argv[i]);
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

				std::string name;
				name = xercesc::XMLString::transcode(element->getNodeName());
				if (name == "xsd:schema")
				{
					parseSchema(element);
				}
			}
		}
		delete parser;
	}

	// find Class that uses an enum
	for (std::list<oscClass *>::iterator cit = classes.begin(); cit != classes.end(); cit++)
	{
		oscClass* cl = *cit;

		std::list<oscMember *> attributes;
		for (std::list<oscMember *>::iterator ait = cl->attributes.begin(); ait != cl->attributes.end(); ait++)
		{
			oscMember *attrib = *ait;
			for (std::list<oscEnum *>::iterator it = enums.begin(); it != enums.end(); it++)
			{
				oscEnum *en = *it;
				if (attrib->type == en->name)
				{
					if (en->ownerClass == NULL)
					{
						en->ownerClass = cl;
					}
					else
					{
						//fprintf(stderr, "enum is used by more than one class %s\n", en->name.c_str());
						cl->enumHeaders.push_back(en->ownerClass->name);
					}
				}
			}
		}
	}


	FILE *registerSchemaObjects = fopen("registerSchemaObjects.h", "w");
	if (registerSchemaObjects == NULL)
	{
		std::cerr << "unable to open registerSchemaObjects.h " << std::endl;
		return -1;
	}
	FILE *schemaHeaders = fopen("schemaHeaders.h", "w");
	if (schemaHeaders == NULL)
	{
		std::cerr << "unable to open schemaHeaders.h " << std::endl;
		return -1;
	}
	FILE *CMakeInc = fopen("oscSchema.inc", "w");
	if (CMakeInc == NULL)
	{
		std::cerr << "unable to open oscSchema.inc " << std::endl;
		return -1;
	}
	FILE *factoriesIncludeHeader = fopen("oscFactoriesHeader.inc", "w");
	if (factoriesIncludeHeader == NULL)
	{
		std::cerr << "unable to open oscFactoriesHeader.inc " << std::endl;
		return -1;
	}
	FILE *factoriesInclude = fopen("oscFactories.inc", "w");
	if (factoriesInclude == NULL)
	{
		std::cerr << "unable to open oscFactories.inc " << std::endl;
		return -1;
	}

	fprintf(CMakeInc, "SET(SCHEMA_HEADERS\n");

	// now write out all classes
	for (std::list<oscClass *>::iterator cit = classes.begin(); cit != classes.end(); cit++)
	{

		oscClass* cl = *cit;

		// find all used classes 
		/*for (std::list<oscMember *>::iterator ait = cl->attributes.begin(); ait != cl->attributes.end(); ait++)
		{
			oscMember *attrib = *ait;
			if (attrib->type.substr(0, 4) == "xsd:")
			{
			}
			else
			{
				bool found = false;
				for (std::list<oscEnum *>::iterator it = enums.begin(); it != enums.end(); it++)
				{
					if ((*it)->name == attrib->type)
					{
						found = true;
						fprintf(header, "    oscEnum %s;\n", attrib->name.c_str());
					}
				}
				if (!found)
				{
					fprintf(stderr, "attribute type %s not implemented or enum not found\n", attrib->type.c_str());
				}
			}
		}*/

		for (std::list<oscMember *>::iterator mit = cl->members.begin(); mit != cl->members.end(); mit++)
		{
			oscMember *member = *mit;
			cl->enumHeaders.push_back(member->type);
		}

		bool foundDuplicate = false;
		for (std::list<oscClass *>::iterator c2it = classes.begin(); c2it != classes.end(); c2it++)
		{
			oscClass* cl2 = *c2it;
			if (cl2 != cl && cl2->name == cl->name)
			{

				fprintf(duplicates, "duplicate class %s, parent %s, class %s, parent %s\n", cl->name.c_str(), cl->parentName.c_str(), cl2->name.c_str(), cl2->parentName.c_str());
				//check if they differ
				if (cl->attributes.size() != cl2->attributes.size())
				{
					fprintf(duplicates, "duplicate class %s differs\n", cl->name.c_str());
					foundDuplicate = true;
					break;
				}
				else
				{
					std::list<oscMember *>::iterator ait = cl->attributes.begin();
					std::list<oscMember *>::iterator ait2 = cl2->attributes.begin();
					for (; ait != cl->attributes.end(); )
					{
						oscMember *attrib = *ait;
						oscMember *attrib2 = *ait2;
						if(attrib->name != attrib2->name)
						{
							fprintf(duplicates, "duplicate class %s differs\n", cl->name.c_str());
							foundDuplicate = true;
							break;
						}
						ait++;
						ait2++;
					}
					if (foundDuplicate)
						break;
				}
				if (cl->members.size() != cl2->members.size())
				{
					fprintf(duplicates, "duplicate class %s differs\n", cl->name.c_str());
					foundDuplicate = true;
					break;
				}
				else
				{
					std::list<oscMember *>::iterator mit = cl->members.begin();
					std::list<oscMember *>::iterator mit2 = cl2->members.begin();
					for (; mit != cl->members.end(); )
					{
						oscMember *attrib = *mit;
						oscMember *attrib2 = *mit2;
						if (attrib->name != attrib2->name)
						{
							fprintf(duplicates, "duplicate class %s differs\n", cl->name.c_str());
							foundDuplicate = true;
							break;
						}
						mit++;
						mit2++;
					}
				}
				if (foundDuplicate)
					break;
				else
				{
					// remove duplicate class
					c2it = classes.erase(c2it);
					if (c2it == classes.end())
						break;
				}
			}
		}
		if (foundDuplicate)
			continue;
		// find enums used by this class and store them in a list
		std::list<oscEnum *> classEnums;
		for (std::list<oscEnum *>::iterator it = enums.begin(); it != enums.end(); it++)
		{
			oscEnum *en = *it;
			if (en->ownerClass == cl)
			{
				classEnums.push_back(en);
			}
		}
		char headerName[1000];
		char cppName[1000];
		char headerDefineName[1000];

		std::string upName(cl->name);
        std::transform(upName.begin(), upName.end(), upName.begin(), ::toupper);

#ifdef WIN32
		_snprintf(headerName, 1000, "schema/%s.h", cl->name.c_str());
		_snprintf(cppName, 1000, "schema/%s.cpp", cl->name.c_str());
		_snprintf(headerDefineName, 1000, "%s_H", upName.c_str());
#else
		snprintf(headerName, 1000, "schema/%s.h", cl->name.c_str());
		snprintf(cppName, 1000, "schema/%s.cpp", cl->name.c_str());
		snprintf(headerDefineName, 1000, "%s_H", upName.c_str());
#endif
		
		FILE *header = fopen(headerName, "w");
		FILE *cpp = NULL;
		if (header == NULL)
		{
			std::cerr << "unable to open " << headerName << std::endl;
			return -1;
		}
		if(classEnums.size()>0)
		{
			cpp = fopen(cppName, "w");
		}

			fprintf(CMakeInc, "schema/%s.h\n",cl->name.c_str());

			fprintf(schemaHeaders, "#include \"schema/%s.h\"\n", cl->name.c_str());
			fprintf(registerSchemaObjects, "staticObjectFactory.registerType<%s>(\"%s\");\n", cl->name.c_str(), cl->name.c_str());
		

		bool catalog = false;
		if (cl->name.find("Catalog") != std::string::npos)
		{
			for (std::list<oscMember *>::iterator mit = cl->members.begin(); mit != cl->members.end(); mit++)
			{
				oscMember *member = *mit;
				if (member->name == "Directory")
				{
					catalog = true;
					break;
				}
			}
		}

				fprintf(header,
			"/* This file is part of COVISE.\n\
\n\
You can use it under the terms of the GNU Lesser General Public License\n\
version 2.1 or later, see lgpl - 2.1.txt.\n\
\n\
* License: LGPL 2 + */\n\
\n\
\n\
#ifndef %s\n\
#define %s\n\
\n\
#include \"../oscExport.h\"\n", headerDefineName, headerDefineName);

		if (catalog)
		{
			fprintf(header, "#include \"../oscCatalog.h\"\n");
		}
		else
		{
			fprintf(header, "#include \"../oscObjectBase.h\"\n");
		}

		fprintf(header,
"#include \"../oscObjectVariable.h\"\n\
#include \"../oscObjectVariableArray.h\"\n\
\n\
#include \"../oscVariables.h\"\n");
		
			for (std::list<std::string>::iterator it = cl->enumHeaders.begin(); it != cl->enumHeaders.end(); it++)
			{
				fprintf(header, "#include \"%s.h\"\n",(*it).c_str());
			}
			fprintf(header, "\nnamespace OpenScenario\n{\n");

		for (std::list<oscEnum *>::iterator it = classEnums.begin(); it != classEnums.end(); it++)
		{
			oscEnum *en = *it;
				fprintf(header, "class OPENSCENARIOEXPORT %sType : public oscEnumType\n{\npublic:\n", en->name.c_str());
				fprintf(header, "\
static %sType *instance();\n\
    private:\n\
		%sType();\n\
	    static %sType *inst; \n\
};\n\
", en->name.c_str(), en->name.c_str(), en->name.c_str());
		}


		if (catalog)
		{
			fprintf(header, "class OPENSCENARIOEXPORT %s : public oscCatalog\n{\npublic:\n\
%s()\n\
{\n\
}\n\
", cl->name.c_str(), cl->name.c_str());

		}
		else
		{
			fprintf(header, "class OPENSCENARIOEXPORT %s : public oscObjectBase\n{\npublic:\n\
%s()\n\
{\n\
", cl->name.c_str(), cl->name.c_str());


		for (std::list<oscMember *>::iterator ait = cl->attributes.begin(); ait != cl->attributes.end(); ait++)
		{
			oscMember *attrib = *ait;

			if(attrib->optional)
			{
				fprintf(header, "        OSC_ADD_MEMBER_OPTIONAL(%s, %d);\n", attrib->name.c_str(), attrib->choice);
			}
			else
			{
				fprintf(header, "        OSC_ADD_MEMBER(%s, %d);\n", attrib->name.c_str(), attrib->choice);
			}
		}
		for (std::list<oscMember *>::iterator mit = cl->members.begin(); mit != cl->members.end(); mit++)
		{
			oscMember *member = *mit;
			
			if (member->optional)
			{
				fprintf(header, "        OSC_OBJECT_ADD_MEMBER_OPTIONAL(%s, \"%s\", %d);\n", member->name.c_str(), member->type.c_str(), member->choice);
			}
			else
			{
				fprintf(header, "        OSC_OBJECT_ADD_MEMBER(%s, \"%s\", %d);\n", member->name.c_str(), member->type.c_str(), member->choice);
			}
		}

		for (std::list<oscMember *>::iterator ait = cl->attributes.begin(); ait != cl->attributes.end(); ait++)
		{
			oscMember *attrib = *ait;
			if (attrib->type.find("Enum_") != std::string::npos)
			{
				bool found = false;
				for (std::list<oscEnum *>::iterator it = enums.begin(); it != enums.end(); it++)
				{
					if ((*it)->name == attrib->type)
					{
						found = true;

						fprintf(header, "        %s.enumType = %sType::instance();\n", attrib->name.c_str(), (*it)->name.c_str());
						break;

						if (!found)
						{
							fprintf(stderr, "attribute type %s not implemented or enum not found\n", attrib->type.c_str());
						}
					}
				}
			}
		}
		
		fprintf(header, "    };\n");

		fprintf(header, "        const char *getScope(){return \"%s\";};\n", cl->parentName.c_str());

		for (std::list<oscMember *>::iterator ait = cl->attributes.begin(); ait != cl->attributes.end(); ait++)
		{
			oscMember *attrib = *ait;
			if (attrib->type == "xsd:string")
				fprintf(header, "    oscString %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:double")
				fprintf(header, "    oscDouble %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:boolean")
				fprintf(header, "    oscBool %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:float")
				fprintf(header, "    oscFloat %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:int")
				fprintf(header, "    oscInt %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:unsignedShort")
				fprintf(header, "    oscUShort %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:unsignedInt")
				fprintf(header, "    oscUInt %s;\n", attrib->name.c_str());
			else if (attrib->type == "xsd:dateTime")
				fprintf(header, "    oscDateTime %s;\n", attrib->name.c_str());
			else
			{
				bool found = false;
				for (std::list<oscEnum *>::iterator it = enums.begin(); it != enums.end(); it++)
				{
					if ((*it)->name == attrib->type)
					{
						found = true;
						fprintf(header, "    oscEnum %s;\n", attrib->name.c_str());
					}
				}
				if (!found)
				{
					fprintf(stderr, "attribute type %s not implemented or enum not found\n", attrib->type.c_str());
				}
			}
		}

		for (std::list<oscMember *>::iterator mit = cl->members.begin(); mit != cl->members.end(); mit++)
		{
			oscMember *member = *mit;
			if (member->array)
			{
				fprintf(header, "    %sArrayMember %s;\n", member->type.c_str(), member->name.c_str());
			}
			else
			{
				fprintf(header, "    %sMember %s;\n", member->type.c_str(), member->name.c_str());
			}
		}
		for (std::list<oscEnum *>::iterator it = classEnums.begin(); it != classEnums.end(); it++)
		{
			oscEnum *en = *it;
			fprintf(header, "\n\
    enum %s\n\
    {\n\
", en->name.c_str());
			for (std::list<std::string>::iterator sit = en->values.begin(); sit != en->values.end(); sit++)
			{
				fprintf(header, "%s,\n", (*sit).c_str());
			}
			fprintf(header, "\n\
    };\n\
");
		}
		}
		
		fprintf(header, "\n\
};\n\n\
typedef oscObjectVariable<%s *> %sMember;\n", cl->name.c_str(), cl->name.c_str());

		fprintf(header, "\
typedef oscObjectVariableArray<%s *> %sArrayMember;\n", cl->name.c_str(), cl->name.c_str());

		// write factoryInc

		fprintf(factoriesInclude, "staticObjectFactory.registerType<%s>(\"%s\");\n", cl->name.c_str(), cl->name.c_str());

		fprintf(factoriesIncludeHeader, "#include \"%s\"\n", cl->name.c_str());


		

		fprintf(header, 
"\n\n\
}\n\n\
#endif //%s\n",headerDefineName);

		if (classEnums.size() > 0)
		{
			// write out cpp


			fprintf(cpp,
				"/* This file is part of COVISE.\n\
\n\
You can use it under the terms of the GNU Lesser General Public License\n\
version 2.1 or later, see lgpl - 2.1.txt.\n\
\n\
* License: LGPL 2 + */\n\
\n\
#include \"%s.h\"\n\
\n\
using namespace OpenScenario;\n\
", cl->name.c_str());

			for (std::list<oscEnum *>::iterator it = classEnums.begin(); it != classEnums.end(); it++)
			{
				currentEnum = *it;

				fprintf(cpp, "%sType::%sType()\n{\n", currentEnum->name.c_str(), currentEnum->name.c_str());

				for (std::list<std::string>::iterator sit = currentEnum->values.begin(); sit != currentEnum->values.end(); sit++)
				{
					fprintf(cpp, "addEnum(\"%s\", %s::%s);\n", (*sit).c_str(),currentEnum->ownerClass->name.c_str(), (*sit).c_str());
				}

				fprintf(cpp, "}\n%sType *%sType::instance()\n\
{\n\
	if (inst == NULL)\n\
	{\n\
		inst = new %sType();\n\
	}\n\
	return inst;\n\
}\n\
", currentEnum->name.c_str(), currentEnum->name.c_str(), currentEnum->name.c_str());
				fprintf(cpp, "%sType *%sType::inst = NULL;\n", currentEnum->name.c_str(), currentEnum->name.c_str());

			}
		}



		fclose(header);
		if(cpp)
		{
			fclose(cpp);
		}
	}
	fclose(duplicates);

	fprintf(CMakeInc, ")\n");

	fprintf(CMakeInc, "SET(SCHEMA_SOURCES\n");
	for (std::list<oscClass *>::iterator cit = classes.begin(); cit != classes.end(); cit++)
	{
		oscClass* cl = *cit;

		// find enums used by this class and store them in a list
		std::list<oscEnum *> classEnums;
		for (std::list<oscEnum *>::iterator it = enums.begin(); it != enums.end(); it++)
		{
			oscEnum *en = *it;
			if (en->ownerClass && cl && en->ownerClass->name == cl->name)
			{
				classEnums.push_back(en);
			}
		}
		if (classEnums.size()>0)
		{
			fprintf(CMakeInc, "schema/%s.cpp\n", cl->name.c_str());
		}
	}
	fprintf(CMakeInc, ")\n");
	fclose(factoriesInclude);
	fclose(factoriesIncludeHeader);
	fclose(CMakeInc);
	fclose(schemaHeaders);
	fclose(registerSchemaObjects);
	
    return 0;
}
