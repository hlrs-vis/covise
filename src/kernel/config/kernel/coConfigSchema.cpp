/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigSchema.h"
#include "coConfigXercesConverter.h"

#include <config/coConfigLog.h>
#include <config/coConfigSchemaInfos.h>
#include <util/string_util.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/psvi/XSModel.hpp>
#include <xercesc/framework/psvi/XSElementDeclaration.hpp>
#include <xercesc/framework/psvi/XSTypeDefinition.hpp>
#include <xercesc/framework/psvi/XSSimpleTypeDefinition.hpp>
#include <xercesc/framework/psvi/XSComplexTypeDefinition.hpp>
#include <xercesc/framework/psvi/XSParticle.hpp>
#include <xercesc/framework/psvi/XSModelGroup.hpp>
#include <xercesc/framework/psvi/XSAnnotation.hpp>
#include <xercesc/framework/psvi/XSConstants.hpp>
#include <xercesc/framework/psvi/XSAttributeUse.hpp>
#include <xercesc/framework/psvi/XSAttributeDeclaration.hpp>
#include <xercesc/framework/psvi/XSNamedMap.hpp>
#include <xercesc/framework/psvi/XSConstants.hpp>
#include <xercesc/framework/psvi/XSNamespaceItem.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/internal/XMLGrammarPoolImpl.hpp>
#endif

#include "coConfigRootErrorHandler.h"

using namespace covise;

covise::coConfigSchema *coConfigSchema::configSchema = 0;
xercesc::XSModel *coConfigSchema::xsModel = 0;
std::string coConfigSchema::fileName = "";

coConfigSchema::~coConfigSchema()
{
    for (const auto &value : elements)
        delete value.second;
}

void coConfigSchema::loadSchema(const std::string &filename)
{
    xercesc::/*coConfig*/ XercesDOMParser *parser;
    coConfigRootErrorHandler handler;
    xercesc::XMLGrammarPool *grammarPool;
    std::string schemaFile = filename;

    if (filename.empty())
    {
        std::string externalSchemaFile = getenv("COCONFIG_SCHEMA");
        if (boost::filesystem::is_regular_file(externalSchemaFile))
        {
            schemaFile = externalSchemaFile;
            COCONFIGDBG("coConfigSchema::loadFile externalSchemaFile: " << schemaFile);
        }
    }
#if XERCES_VERSION_MAJOR < 3
    grammarPool = new xercesc::XMLGrammarPoolImpl(xercesc::XMLPlatformUtils::fgMemoryManager);
#else
    grammarPool = NULL;
#endif
    parser = new xercesc::/*coConfig*/ XercesDOMParser(0, xercesc::XMLPlatformUtils::fgMemoryManager, grammarPool);
    // parser->setExternalNoNamespaceSchemaLocation( schemaFile.utf16() );
    parser->setDoNamespaces(true); // n o change
    parser->setDoSchema(true);
    parser->useCachedGrammarInParse(false);
    // parser->setDoValidation( true );
    parser->setValidationSchemaFullChecking(true);
    parser->setIncludeIgnorableWhitespace(false);
    //m! changed from never
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);
    parser->setErrorHandler(&handler);
    parser->loadGrammar(stringToXexcesc(schemaFile).get(), xercesc::Grammar::SchemaGrammarType, true);
    if (handler.getSawErrors())
    {
        handler.resetErrors();
        //  Try parsing fallback schema file
        std::string fallbackSchema = getenv("COVISEDIR");
        fallbackSchema.append("/src/kernel/config/coEditor/schema.xsd");
        COCONFIGLOG("coConfigSchema::loadFile err: Parsing '" << schemaFile << "' failed. Trying fallback:" << fallbackSchema);
        parser->loadGrammar(stringToXexcesc(fallbackSchema).get(), xercesc::Grammar::SchemaGrammarType, true);
    }

    if (!handler.getSawErrors())
    {
//create an instance of coConfigSchema, all data of schema can be accessed there
//       xsModel = grammarPool->getXSModel();

#if XERCES_VERSION_MAJOR < 3
        //TODO change validation once we have a valid schema
        delete coConfigSchema::getInstance();
        coConfigSchema::getInstance(grammarPool->getXSModel());
        fileName = filename;
#endif
    }
    else
    {
        COCONFIGLOG("coConfigSchema::loadFile warn: Parse of Schema failed "
                    << "\n");
    }
}

void coConfigSchema::walkTree()
{
    if (walked)
        return;
    {
        const XMLCh *compNamespace = xercesc::XMLString::transcode("");
        const XMLCh *coConfigTag = xercesc::XMLString::transcode("COCONFIG");
        coConfigSchemaInfos *coconfigInfo = createSchemaInfos(xsModel->getElementDeclaration(coConfigTag, compNamespace));
        if (coconfigInfo)
            elements.insert({"COCONFIG", coconfigInfo});
        sortInGroups();
        walked = 1;
    }
}

// sort the coConfigSchemaInfos by their group into groupLists
void coConfigSchema::sortInGroups()
{
    if (elements.empty())
        return;

    for (const auto &element : elements)
    {
        if (element.second)
        {
            std::string elementGroup = element.second->getElementGroup();
            if (!elementGroup.empty()) // only schemaInfos with an element group are added
            {
                // Add item to All list
                if (groups.find(elementGroup) != groups.end())
                {
                    coConfigSchemaInfosList &groupList = groups[elementGroup];
                    groupList.insert(element.second);
                }
                else
                {
                    coConfigSchemaInfosList groupList{element.second};
                    groups.insert({elementGroup, groupList});
                }
            }
        }
    }
}

/// walk the SchemaTree from a given Element, create coConfigSchemaInfos for each and fill them with name in elements Hash
coConfigSchemaInfos *coConfigSchema::createSchemaInfos(xercesc::XSElementDeclaration *elementDecl, const std::string parent)
{

    if (!elementDecl)
    {
        return 0;
    }

    coConfigSchemaInfos *schemaInfos = new coConfigSchemaInfos();
    std::string name = xercescToStdString(elementDecl->getName());
    schemaInfos->setElement(name);
    schemaInfos->setElementPath(parent);

    addAnnotationsForElement(elementDecl, schemaInfos);

    ///process elementType
    xercesc::XSTypeDefinition *typeDef = elementDecl->getTypeDefinition();
    if (typeDef)
    {
        if (typeDef->getTypeCategory() == xercesc::XSTypeDefinition::COMPLEX_TYPE)
        {
            xercesc::XSComplexTypeDefinition *complexTypeDef = static_cast<xercesc::XSComplexTypeDefinition *>(typeDef);
            ///process Attributes of type
            xercesc::XSAttributeUseList *attributeUses = complexTypeDef->getAttributeUses();
            if (attributeUses != NULL)
            {
                xercesc::XSAttributeUse *attrUse;
                for (unsigned int i = 0; i < attributeUses->size(); i++)
                {
                    attrUse = attributeUses->elementAt(i);
                    xercesc::XSAttributeDeclaration *attrDecl = (attrUse->getAttrDeclaration());
                    if (attrDecl != NULL)
                    {
                        // get constraint defaultvalue (fixed constraint not handled)
                        xercesc::XSConstants::VALUE_CONSTRAINT valueConstraintType = attrUse->getConstraintType();
                        std::string defValue = "";
                        if (valueConstraintType == xercesc::XSConstants::VALUE_CONSTRAINT_DEFAULT)
                        {
                            defValue = xercescToStdString(attrUse->getConstraintValue());
                        }
                        //get type of attribute
                        xercesc::XSSimpleTypeDefinition *type = attrDecl->getTypeDefinition();
                        auto annos = createFromSchemaFileAnnotationsList(type->getAnnotations());
                        std::string readableRule = annos["attributeTypeInfo"];
                        ///   now finaly add infos for one attribute to cConfigSchemaInfo object
                        ///   Str name of attribute e.g.  value
                        ///   bool if attribute is required or optional
                        ///   Str defaultValue
                        ///   Str readableRule
                        ///   get std::string pattern for RegExp
                        ///   Str attributeDescrition
                        schemaInfos->addAttribute(xercescToStdString(attrDecl->getName()), attrUse->getRequired(), defValue, readableRule, processSimpleTypeDefinition(type) /*, attributeDescrition*/);
                    }
                }
            }
            ///end process Attributes of type,
            // check contentType to see if children are allowed
            xercesc::XSComplexTypeDefinition::CONTENT_TYPE contentType = complexTypeDef->getContentType();
            if (contentType == xercesc::XSComplexTypeDefinition::CONTENTTYPE_ELEMENT || contentType == xercesc::XSComplexTypeDefinition::CONTENTTYPE_MIXED)
            {
                // get Particle, i.e. allowed children
                auto allowedChildren = processParticle(complexTypeDef->getParticle());
                if (!allowedChildren.empty())
                {
                    std::string childPath = parent + "." + name;
                    for (const auto allowedChild : allowedChildren)
                    {
                        coConfigSchemaInfos *childInfos = createSchemaInfos(allowedChild, childPath);
                        // elements.insert( std::string::fromUtf16((*iter)->getName()), childInfos);

                        if (childInfos)
                        {
                            std::string id = childInfos->getElementPath() + "." + childInfos->getElement();
                            elements.insert({id, childInfos});
                        }
                    }
                }
            }
        }
    }

    return schemaInfos;
}

/// get annotations for a element and fill them into the coConfigSchemaInfos
void coConfigSchema::addAnnotationsForElement(xercesc::XSElementDeclaration *elementDecl, coConfigSchemaInfos *schemaInfos)
{
    // fetch annotations
    auto annos = createFromSchemaFileAnnotation(elementDecl->getAnnotation());
    //(i.e. a std::map<std::string, std::string*>* with e.g. "elementGroup", "VRML")
    // fill the annotations in the coConfigSchemaInfos
    schemaInfos->setElementGroup(annos["elementGroup"]);
    schemaInfos->setElementDescription(annos["elementDescription"]);
    schemaInfos->setElementName(annos["elementName"]);
    schemaInfos->setReadableElementRule(annos["readableElementRule"]);
}

std::string coConfigSchema::processSimpleTypeDefinition(xercesc::XSSimpleTypeDefinition *xsSimpleTypeDef)
{
    std::string pattern = "";

    int facets = xsSimpleTypeDef->getDefinedFacets();
    if (facets)
    {
        if (facets & xercesc::XSSimpleTypeDefinition::FACET_PATTERN)
        {
            xercesc::StringList *lexicalPatterns = xsSimpleTypeDef->getLexicalPattern();
            // it divides the pattern by | which is not good for RegExp, thou i repair it.
            if (lexicalPatterns && lexicalPatterns->size())
            {
                for (unsigned i = 0; i < lexicalPatterns->size(); i++)
                {
                    pattern.append(xercescToStdString(lexicalPatterns->elementAt(i)));
                    pattern.append("|");
                }
                auto s = split(pattern, '|');
                pattern.clear();
                for (size_t i = 0; i < s.size() - 2; i++)
                {
                    pattern += s[i];
                }
            }
        }
    }
    return pattern;
}

std::map<std::string, std::string> coConfigSchema::createFromSchemaFileAnnotationsList(xercesc::XSAnnotationList *annoList)
{
    std::map<std::string, std::string> schemaInfos;
    if (!annoList)
    {
        return schemaInfos;
    }
    for (unsigned int i = 0; i < annoList->size(); ++i)
    {
        //assign hashes to one
        auto sis = createFromSchemaFileAnnotation(annoList->elementAt(i));
        schemaInfos.insert(sis.begin(), sis.end());
    }
    return schemaInfos;
}

/// collects content of xs:appinfo (e.g.: name = "MyName"), divides them by = and adds to a std::map
std::map<std::string, std::string> coConfigSchema::createFromSchemaFileAnnotation(xercesc::XSAnnotation *anno)
{
    std::map<std::string, std::string> schemaInfos;
    if (!anno)
    {
        return schemaInfos;
    }

    std::string annotation = xercescToStdString(anno->getAnnotationString());
    std::regex rx("</?xs:appinfo.*?>");
    auto annos = split(annotation, rx, true);
    for (const auto iter : annos)
    {
        if (!strip(iter).empty() && iter.find("<") == std::string::npos)
        {
            auto s = split(iter, '\"');
            assert(s.size() > 1);
            schemaInfos.insert(std::make_pair(strip(iter.substr(0, iter.find("="))), s[1]));
        }
    }
    return schemaInfos;
}

// check compositor type of elements. Get allowed children
std::vector<xercesc::XSElementDeclaration *> coConfigSchema::processParticle(xercesc::XSParticle *xsParticle)
{

    std::vector<xercesc::XSElementDeclaration *> childrenParticleList;
    if (xsParticle)
    {
        xercesc::XSParticle::TERM_TYPE termType = xsParticle->getTermType();
        if (termType == xercesc::XSParticle::TERM_ELEMENT)
        {
            childrenParticleList.push_back(xsParticle->getElementTerm());
            return childrenParticleList;
        }
        else if (termType == xercesc::XSParticle::TERM_MODELGROUP)
        {
            xercesc::XSModelGroup *xsModelGroup = xsParticle->getModelGroupTerm();
            xercesc::XSParticleList *xsParticleList = xsModelGroup->getParticles();
            for (unsigned i = 0; i < xsParticleList->size(); i++)
            {
                auto pp = processParticle(xsParticleList->elementAt(i));
                childrenParticleList.insert(childrenParticleList.end(), pp.begin(), pp.end());
            }
            return childrenParticleList;
        }
        else if (termType == xercesc::XSParticle::TERM_WILDCARD)
        {
            return childrenParticleList;
        }
    }
    return childrenParticleList;
}

// NOTE unused atm
std::string coConfigSchema::printCompositorTypeConnector(xercesc::XSModelGroup::COMPOSITOR_TYPE type)
{
    switch (type)
    {
    //i.e. ,
    case xercesc::XSModelGroup::COMPOSITOR_SEQUENCE:
        return std::string("COMPOSITOR_SEQUENCE");
        break;
    //i.e. |
    case xercesc::XSModelGroup::COMPOSITOR_CHOICE:
        return std::string("COMPOSITOR_CHOICE");
        break;
    case xercesc::XSModelGroup::COMPOSITOR_ALL: //i.e. *
        return std::string("COMPOSITOR_ALL");
        break;
    }
    return std::string();
}

/// return a List of all Groups (is used in mainwindow to create treeModel)
std::set<std::string> coConfigSchema::getGroupsFromSchema()
{
    std::set<std::string> rv;
    for (const auto &g : groups)
        rv.insert(g.first);
    return rv;
}

/// deliver coConfigSchemaInfosList for group if exists, else nullPointer
coConfigSchemaInfosList *coConfigSchema::getSchemaInfosForGroup(const std::string groupName)
{
    if (!groupName.empty() && walked)
    {
        auto groupList = groups.find(groupName);
        if (groupList != groups.end())
            return &groupList->second;
    }
    return nullptr;
}

/// return a pointer to the coConfigSchemaInfos of an element
coConfigSchemaInfos *coConfigSchema::getSchemaInfosForElement(const std::string &name)
{
    if (!name.empty() && walked)
    {
        auto el = elements.find(name);
        if (el != elements.end())
            return el->second;
    }
    return nullptr;
}
