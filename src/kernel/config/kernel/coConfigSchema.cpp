/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigSchema.h"
#include <config/coConfigSchemaInfos.h>
#include <config/coConfigSchemaInfosList.h>
#include <config/coConfigLog.h>
#include <QFileInfo>

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
QString coConfigSchema::fileName = "";

coConfigSchema::~coConfigSchema()
{
    foreach (coConfigSchemaInfos *value, elements)
        delete value;
}

void coConfigSchema::loadSchema(const QString &filename)
{
    xercesc::/*coConfig*/ XercesDOMParser *parser;
    coConfigRootErrorHandler handler;
    xercesc::XMLGrammarPool *grammarPool;
    QString schemaFile = filename;

    if (filename.isEmpty())
    {
        QString externalSchemaFile = getenv("COCONFIG_SCHEMA");
        if (/*QFileInfo(externalSchemaFile).exists() &&*/ QFileInfo(externalSchemaFile).isFile())
        {
            schemaFile = externalSchemaFile;
            COCONFIGDBG("coConfigSchema::loadFile externalSchemaFile: " << schemaFile.toLatin1());
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
    parser->loadGrammar(reinterpret_cast<const XMLCh *>(schemaFile.utf16()), xercesc::Grammar::SchemaGrammarType, true);
    if (handler.getSawErrors())
    {
        handler.resetErrors();
        //  Try parsing fallback schema file
        QString fallbackSchema = getenv("COVISEDIR");
        fallbackSchema.append("/src/kernel/config/coEditor/schema.xsd");
        COCONFIGLOG("coConfigSchema::loadFile err: Parsing '" << schemaFile << "' failed. Trying fallback:" << fallbackSchema);
        parser->loadGrammar(reinterpret_cast<const XMLCh *>(fallbackSchema.utf16()), xercesc::Grammar::SchemaGrammarType, true);
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
            elements.insert("COCONFIG", coconfigInfo);
        sortInGroups();
        walked = 1;
    }
}

// sort the coConfigSchemaInfos by their group into groupLists
void coConfigSchema::sortInGroups()
{
    if (elements.isEmpty())
        return;

    //m
    coConfigSchemaInfosList *allGroupList = new coConfigSchemaInfosList();

    for (QHash<QString, coConfigSchemaInfos *>::const_iterator iter = elements.begin(); iter != elements.end(); ++iter)
    {
        if (iter.value())
        {
            //COCONFIGLOG ("coConfigSchema::sort info: sorting element " << (*iter)->getElement()  );
            //sort this element into a coConfigSchemaInfosList, all Lists are in the groups QHash
            QString elementGroup = iter.value()->getElementGroup();
            if (!elementGroup.isEmpty()) // only schemaInfos with an element group are added
            {
                // Add item to All list
                if (!allGroupList->contains(iter.value()))
                    allGroupList->append(iter.value());
                if (groups.contains(elementGroup))
                {
                    coConfigSchemaInfosList *groupList = groups.value(elementGroup);
                    if (!groupList->contains(iter.value()))
                    {
                        groupList->append(iter.value());
                    }
                }
                else
                {
                    coConfigSchemaInfosList *groupList = new coConfigSchemaInfosList();
                    groupList->append(iter.value());
                    groups.insert(elementGroup, groupList);
                }
            }
        }
    }
}

/// walk the SchemaTree from a given Element, create coConfigSchemaInfos for each and fill them with name in elements Hash
coConfigSchemaInfos *coConfigSchema::createSchemaInfos(xercesc::XSElementDeclaration *elementDecl, const QString parent)
{

    if (!elementDecl)
    {
        return 0;
    }

    coConfigSchemaInfos *schemaInfos = new coConfigSchemaInfos();
    QString name = QString::fromUtf16(reinterpret_cast<const ushort *>(elementDecl->getName()));
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
                        QString defValue = "";
                        if (valueConstraintType == xercesc::XSConstants::VALUE_CONSTRAINT_DEFAULT)
                        {
                            defValue = QString::fromUtf16(reinterpret_cast<const ushort *>(attrUse->getConstraintValue()));
                        }
                        //get type of attribute
                        xercesc::XSSimpleTypeDefinition *type = attrDecl->getTypeDefinition();
                        QHash<QString, QString> *annos = createFromSchemaFileAnnotationsList(type->getAnnotations());
                        QString readableRule = annos->value("attributeTypeInfo");
                        ///   now finaly add infos for one attribute to cConfigSchemaInfo object
                        ///   QStr name of attribute e.g.  value
                        ///   bool if attribute is required or optional
                        ///   QStr defaultValue
                        ///   QStr readableRule
                        ///   get QString pattern for RegExp
                        ///   QStr attributeDescrition
                        schemaInfos->addAttribute(QString::fromUtf16(reinterpret_cast<const ushort *>(attrDecl->getName())), attrUse->getRequired(), defValue, readableRule, processSimpleTypeDefinition(type) /*, attributeDescrition*/);
                    }
                }
            }
            ///end process Attributes of type,
            // check contentType to see if children are allowed
            xercesc::XSComplexTypeDefinition::CONTENT_TYPE contentType = complexTypeDef->getContentType();
            if (contentType == xercesc::XSComplexTypeDefinition::CONTENTTYPE_ELEMENT || contentType == xercesc::XSComplexTypeDefinition::CONTENTTYPE_MIXED)
            {
                // get Particle, i.e. allowed children
                QList<xercesc::XSElementDeclaration *> allowedChildren = processParticle(complexTypeDef->getParticle());
                if (!allowedChildren.isEmpty())
                {
                    QString childPath = parent + "." + name;
                    for (QList<xercesc::XSElementDeclaration *>::const_iterator iter = allowedChildren.begin(); iter != allowedChildren.end(); ++iter)
                    {
                        coConfigSchemaInfos *childInfos = createSchemaInfos((*iter), childPath);
                        //elements.insert( QString::fromUtf16((*iter)->getName()), childInfos);

                        if (childInfos)
                        {
                            QString id = childInfos->getElementPath() + "." + childInfos->getElement();
                            elements.insert(id, childInfos);
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
    QHash<QString, QString> *annos = createFromSchemaFileAnnotation(elementDecl->getAnnotation());
    //(i.e. a QHash<QString, QString*>* with e.g. "elementGroup", "VRML")
    //fill the annotations in the coConfigSchemaInfos
    schemaInfos->setElementGroup(annos->value("elementGroup"));
    schemaInfos->setElementDescription(annos->value("elementDescription"));
    schemaInfos->setElementName(annos->value("elementName"));
    schemaInfos->setReadableElementRule(annos->value("readableElementRule"));
}

QString coConfigSchema::processSimpleTypeDefinition(xercesc::XSSimpleTypeDefinition *xsSimpleTypeDef)
{
    QString pattern = "";

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
                    pattern.append(QString::fromUtf16(reinterpret_cast<const ushort *>(lexicalPatterns->elementAt(i))));
                    pattern.append("|");
                }
                pattern = pattern.section("|", 0, -2);
            }
        }
    }
    return pattern;
}

QHash<QString, QString> *coConfigSchema::createFromSchemaFileAnnotationsList(xercesc::XSAnnotationList *annoList)
{
    QHash<QString, QString> *schemaInfos = new QHash<QString, QString>;
    if (!annoList)
    {
        return schemaInfos;
    }
    for (unsigned int i = 0; i < annoList->size(); ++i)
    {
        //assign hashes to one
        schemaInfos = createFromSchemaFileAnnotation(annoList->elementAt(i));
    }
    return schemaInfos;
}

/// collects content of xs:appinfo (e.g.: name = "MyName"), divides them by = and adds to a QHash
QHash<QString, QString> *coConfigSchema::createFromSchemaFileAnnotation(xercesc::XSAnnotation *anno)
{
    QHash<QString, QString> *schemaInfos = new QHash<QString, QString>;
    if (!anno)
    {
        return schemaInfos;
    }

    QString annotation = QString::fromUtf16(reinterpret_cast<const ushort *>(anno->getAnnotationString()));
    QRegExp rx("</?xs:appinfo.*>");
    rx.setMinimal(1);
    QStringList annos = annotation.split(rx);
    for (QStringList::const_iterator iter = annos.begin(); iter != annos.end(); ++iter)
    {
        if (!(*iter).trimmed().isEmpty() && !(*iter).contains("<"))
        {
            schemaInfos->insert((*iter).section("=", 0, 0).trimmed(), (*iter).section("\"", 1, 1).trimmed());
            //COCONFIGLOG ( ( *iter ).trimmed().section ("=", 0, 0) << "\t\t" << ( *iter ).trimmed().section ("\"", 1, 1) );
        }
    }
    return schemaInfos;
}

// check compositor type of elements. Get allowed children
QList<xercesc::XSElementDeclaration *> coConfigSchema::processParticle(xercesc::XSParticle *xsParticle)
{

    QList<xercesc::XSElementDeclaration *> childrenParticleList;
    if (xsParticle)
    {
        xercesc::XSParticle::TERM_TYPE termType = xsParticle->getTermType();
        if (termType == xercesc::XSParticle::TERM_ELEMENT)
        {
            xercesc::XSElementDeclaration *xsElement = xsParticle->getElementTerm();
            QString child = QString::fromUtf16(reinterpret_cast<const ushort *>(xsElement->getName()));
            childrenParticleList.append(xsElement);
            return childrenParticleList;
        }
        else if (termType == xercesc::XSParticle::TERM_MODELGROUP)
        {
            xercesc::XSModelGroup *xsModelGroup = xsParticle->getModelGroupTerm();
            //xercesc::XSModelGroup::COMPOSITOR_TYPE compositorType = xsModelGroup->getCompositor();
            xercesc::XSParticleList *xsParticleList = xsModelGroup->getParticles();
            for (unsigned i = 0; i < xsParticleList->size(); i++)
            {
                childrenParticleList << (processParticle(xsParticleList->elementAt(i)));
                //              if(xsParticleList->elementAt(i)->getTermType() == xercesc::XSParticle::TERM_ELEMENT )
                //childrenParticleList.append( printCompositorTypeConnector (compositorType) );
            }
            //childrenParticleList << processParticle(xsParticleList->elementAt (xsParticleList->size() - 1) );
            //children.append ( printCompositorTypeConnector (compositorType) );
            return childrenParticleList;
        }
        else if (termType == xercesc::XSParticle::TERM_WILDCARD)
        {
            // COCONFIGLOG ( "* (wildcard)");
            //children.append ("COMPOSITOR_ALL");
            return childrenParticleList;
        }
    }
    return childrenParticleList;
}

// NOTE unused atm
QString coConfigSchema::printCompositorTypeConnector(xercesc::XSModelGroup::COMPOSITOR_TYPE type)
{
    switch (type)
    {
    //i.e. ,
    case xercesc::XSModelGroup::COMPOSITOR_SEQUENCE:
        return QString("COMPOSITOR_SEQUENCE");
        break;
    //i.e. |
    case xercesc::XSModelGroup::COMPOSITOR_CHOICE:
        return QString("COMPOSITOR_CHOICE");
        break;
    case xercesc::XSModelGroup::COMPOSITOR_ALL: //i.e. *
        return QString("COMPOSITOR_ALL");
        break;
    }
    return QString::null;
}

/// return a List of all Groups (is used in mainwindow to create treeModel)
QStringList coConfigSchema::getGroupsFromSchema()
{
    if (!groups.isEmpty())
    {
        QStringList groupsList = groups.keys();
        return static_cast<QStringList>(groupsList);
    }
    else
        return QStringList();
}

/// deliver coConfigSchemaInfosList for group if exists, else nullPointer
coConfigSchemaInfosList *coConfigSchema::getSchemaInfosForGroup(const QString groupName)
{
    if (!groupName.isEmpty() && walked)
    {
        coConfigSchemaInfosList *groupList = groups.value(groupName, 0);
        return groupList;
    }
    else
    {
        return 0;
    }
}

/// return a pointer to the coConfigSchemaInfos of an element
coConfigSchemaInfos *coConfigSchema::getSchemaInfosForElement(const QString &name)
{
    if (!name.isEmpty() && walked)
    {
        return elements.value(name, 0);
    }
    else
    {
        return 0;
    }
}
