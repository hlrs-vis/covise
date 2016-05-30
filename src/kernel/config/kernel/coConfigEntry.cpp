/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QRegExp>
#include <QLinkedList>

#include <xercesc/dom/DOM.hpp>

#include <config/coConfigLog.h>
#include <config/coConfigSchemaInfos.h>
#include "coConfigSchema.h"
#include "coConfigXercesEntry.h"
#include "coConfigTools.h"

using namespace covise;

// #include <config/coConfigEvent.h>

coConfigXercesEntry::coConfigXercesEntry()
{
}

coConfigXercesEntry::~coConfigXercesEntry()
{
}

coConfigEntry::coConfigEntry()
    : isListNode(false)
    , readOnly(false)
    , cName(0)
    , name(QString::null)
{
    schemaInfos = 0;
}

coConfigEntry::coConfigEntry(const coConfigEntry *source)
    : configScope(source->configScope)
    , configName(source->configName)
    , path(source->path)
    , isListNode(source->isListNode)
    , readOnly(source->readOnly)
    , textNodes(source->textNodes)
    , schemaInfos(source->schemaInfos)
    , elementGroup(source->elementGroup)
    , cName(0)
    , name(source->name)
{
    for (coConfigEntryPtrList::const_iterator child = source->children.begin();
         child != source->children.end(); ++child)
    {
        coConfigEntry *childClone = (*child)->clone();
        if (childClone)
            this->children.append(childClone);
        else
            COCONFIGDBG("coConfigEntry::<init> err: child clone returned 0");
    }

    QList<QString> keys = source->attributes.keys();

    for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
    {
        if (source->attributes[*key] == 0)
            this->attributes[*key] = 0;
        else
            this->attributes[*key] = new QString(*(source->attributes[*key]));
    }
}

coConfigXercesEntry::coConfigXercesEntry(const coConfigXercesEntry *source)
    : coConfigEntry(source)
{
}

coConfigEntry::~coConfigEntry()
{
    //    notify();
    QList<QString> keys = attributes.keys();
    for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
    {
        delete attributes.take(*key);
    }
}

void coConfigEntry::entryChanged()
{
    //COCONFIGDBG("coConfigEntry::entryChanged(): info: notify called  " << this->getName());
    notify();
}

bool coConfigEntry::matchingAttributes() const
{
    return coConfigTools::matchingAttributes(attributes);
}

xercesc::DOMNode *coConfigXercesEntry::storeToDom(xercesc::DOMDocument &document, int indent)
{

    xercesc::DOMElement *element = document.createElement(reinterpret_cast<const XMLCh *>(path.section('.', -1).utf16()));

    if (xercesc::DOMNode::ELEMENT_NODE == element->getNodeType())
    {
        bool include = QString::fromUtf16(reinterpret_cast<const ushort *>(element->getNodeName())) == "INCLUDE";
        bool tryinclude = QString::fromUtf16(reinterpret_cast<const ushort *>(element->getNodeName())) == "TRYINCLUDE";
        if (include || tryinclude)
        {
            if (isListNode)
            {
                QList<QString> keys = attributes.keys();

                for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
                {
                    element->setAttribute(reinterpret_cast<const XMLCh *>((*key).utf16()), reinterpret_cast<const XMLCh *>((*attributes[*key]).utf16()));
                    COCONFIGDBG("coConfigEntry::storeToDom info: includenode");
                }
                element->appendChild(document.createTextNode(xercesc::XMLString::transcode("\n  ")));
                QStringList values;
                for (QStringList::iterator it = textNodes.begin(); it != textNodes.end(); ++it)
                {
                    values.append(*it);
                }
                element->appendChild(document.createTextNode(reinterpret_cast<const XMLCh *>(values.join("\n  ").utf16())));
            }
            element->appendChild(document.createTextNode(xercesc::XMLString::transcode("\n")));
        }
        else
        {
            QList<QString> keys = attributes.keys();

            for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
            {
                if (attributes[*key])
                {
                    element->setAttribute(reinterpret_cast<const XMLCh *>((*key).utf16()), reinterpret_cast<const XMLCh *>((*attributes[*key]).utf16()));
                    COCONFIGDBG("coConfigEntry::storeToDom info: noIncludenode key " << (*key) << " attr " << (*attributes[*key]));
                }
            }

            if (isListNode)
            {
                QStringList values;
                for (QStringList::iterator it = textNodes.begin(); it != textNodes.end(); ++it)
                {
                    values.append(*it);
                    COCONFIGDBG("coConfigEntry::storeToDom info: listnode " << (*it));
                }
                element->appendChild(document.createTextNode(reinterpret_cast<const XMLCh *>(values.join("\n" + QString().fill(' ', indent)).utf16())));
            }

            for (coConfigEntryPtrList::iterator entry = children.begin();
                 entry != children.end(); ++entry)
            {
                COCONFIGDBG("coConfigEntry::storeToDom info: children " << (*entry)->getName());
                element->appendChild(document.createTextNode(reinterpret_cast<const XMLCh *>(QString("\n" + QString().fill(' ', indent)).utf16())));
                coConfigXercesEntry *xentry = dynamic_cast<coConfigXercesEntry *>(*entry);
                if (xentry)
                    element->appendChild((xentry)->storeToDom(document, indent + 2));
                COCONFIGDBG("coConfigEntry::storeToDom info: childexit " << (*entry)->getName());
            }

            if (indent > 2)
                element->appendChild(document.createTextNode(reinterpret_cast<const XMLCh *>(QString("\n" + QString().fill(' ', indent - 2)).utf16())));
            else
                element->appendChild(document.createTextNode(xercesc::XMLString::transcode("\n")));
        }
    }

    return element;
}

coConfigEntry *coConfigXercesEntry::restoreFromDom(xercesc::DOMElement *node,
                                                   const QString &configName)
{
    coConfigEntry *entry = new coConfigXercesEntry();
    QString qNodeName = QString::fromUtf16(reinterpret_cast<const ushort *>(node->getNodeName()));
    QString path = qNodeName;
    entry->configName = configName;

    if (configName.isNull())
    {
        COCONFIGLOG("coConfigEntry::restoreFromDom warn: no config name: ");
    }

    // Tree upward iteration to get scope attribute and complete path ,if not already one step before yac
    xercesc::DOMNode *scopeNode = node;

    if (node->getParentNode() == 0 && QString::fromUtf16(reinterpret_cast<const ushort *>(node->getNodeName())).toUpper() != "COCONFIG" && node->getNodeType() != xercesc::DOMNode::DOCUMENT_NODE)
    {
    }
    else
    { //iterate parents
        for (xercesc::DOMNode *parent = node->getParentNode();
             parent != 0 && QString::fromUtf16(reinterpret_cast<const ushort *>(parent->getNodeName())).toUpper() != "COCONFIG" && parent->getNodeType() != xercesc::DOMNode::DOCUMENT_NODE;
             parent = parent->getParentNode())
        {
            path = QString::fromUtf16(reinterpret_cast<const ushort *>(parent->getNodeName())) + "." + path;
            scopeNode = parent;
        }
    }

    XMLCh *scopeTag = xercesc::XMLString::transcode("scope");

    if (xercesc::DOMNode::ELEMENT_NODE == scopeNode->getNodeType())
    {
        xercesc::DOMElement *scopeElement = static_cast<xercesc::DOMElement *>(scopeNode);
        QString configScope = QString::fromUtf16(reinterpret_cast<const ushort *>(scopeElement->getAttribute(scopeTag)));
        if (QString::fromUtf16(reinterpret_cast<const ushort *>(scopeElement->getNodeName())).toUpper() == "GLOBAL")
        {
            entry->configScope = coConfigConstants::Global;
        }
        else if (QString::fromUtf16(reinterpret_cast<const ushort *>(scopeElement->getNodeName())).toUpper() == "LOCAL")
        {
            entry->configScope = coConfigConstants::Host;
        }
        else if (QString::fromUtf16(reinterpret_cast<const ushort *>(scopeElement->getNodeName())).toUpper() == "CLUSTER")
        {
            entry->configScope = coConfigConstants::Cluster;
        }
        else if (!configScope.isNull())
        {
            COCONFIGLOG("coConfigEntry::restoreFromDom warn: unknown config scope <" << configScope << ">");
        }
    }
    xercesc::XMLString::release(&scopeTag);
    entry->setPath(path);

    // get node attributes and their values
    xercesc::DOMNamedNodeMap *map = node->getAttributes();
    for (int ctr = 0; ctr < map->getLength(); ++ctr)
    {
        QString nodeName = QString::fromUtf16(reinterpret_cast<const ushort *>(map->item(ctr)->getNodeName()));
        QString nodeValue = QString::fromUtf16(reinterpret_cast<const ushort *>(map->item(ctr)->getNodeValue()));
        entry->attributes.insert(nodeName, new QString(nodeValue));
    }

    // collect all child nodes. if text, add. if element start recursition
    xercesc::DOMNodeList *nodeList = node->getChildNodes();
    for (int i = 0; i < nodeList->getLength(); ++i)
    {
        //if textNode, not empty line, it is added to entry
        if (nodeList->item(i)->getNodeType() == xercesc::DOMNode::TEXT_NODE)
        {
            QString nodeValue = QString::fromUtf16(reinterpret_cast<const ushort *>(nodeList->item(i)->getNodeValue()));
            if (!nodeValue.trimmed().isEmpty())
            {
                entry->isListNode = true;
                QStringList textlist = nodeValue.split("\n", QString::SkipEmptyParts);
                for (QStringList::iterator i = textlist.begin(); i != textlist.end(); ++i)
                {
                    QString singleNodeValue = (*i).trimmed();
                    if (!singleNodeValue.isEmpty())
                    {
                        entry->attributes.insert(*i, new QString());
                        entry->textNodes.append(*i);
                    }
                }
            }
        }
        if (xercesc::DOMNode::ELEMENT_NODE == nodeList->item(i)->getNodeType())
        {
            if (coConfigXercesEntry *xentry = dynamic_cast<coConfigXercesEntry *>(entry))
                xentry->children.append(coConfigXercesEntry::restoreFromDom(static_cast<xercesc::DOMElement *>(nodeList->item(i)), configName));
        }
    }
    //COCONFIGLOG( "coConfigEntry::restoreFromDom info: new entry : "<< path );
    //FIXME still needed?    entry->setElementGroup(path);
    return entry;
}

void coConfigEntry::merge(const coConfigEntry *with)
{
    //COCONFIGLOG("coConfigEntry::merge info: " << getCName());

    if (isReadOnly())
    {
        COCONFIGLOG("coConfigEntry::merge warn: trying to merge read only entry");
        return;
    }

    for (coConfigEntryPtrList::const_iterator entry = with->children.begin();
         entry != with->children.end(); ++entry)
    {
        coConfigEntryPtrList::iterator myEntry = children.begin();
        for (; myEntry != children.end(); ++myEntry)
        {
            if ((*entry)->getName() == (*myEntry)->getName())
            {
                (*myEntry)->merge((*entry));
                break;
            }
        }

        if (myEntry == children.end())
            children.append((*entry)->clone());
    }

    // FIXME Merge attributes
    QList<QString> keys = attributes.keys();
    QList<QString> withKeys = with->attributes.keys();

    for (QList<QString>::iterator withKey = withKeys.begin(); withKey != withKeys.end(); ++withKey)
    {
        if (keys.contains(*withKey))
        {
            //          COCONFIGDBG("coConfigEntry::merge warn: duplicate attribute " << withKey->toLatin1().data() << ", ignoring");
        }
        else
        {
            COCONFIGDBG("coConfigEntry::merge info: merging attribute " << withKey->toLatin1().data());
            attributes.insert(*withKey, with->attributes[*withKey]);
        }
    }
}

// coConfigEntry * coConfigEntry::clone() const
// {
//    if (this == 0)
//       return 0;
//    else
//       return new coConfigEntry(this);
// }

coConfigEntry *coConfigXercesEntry::clone() const
{
    return new coConfigXercesEntry(this);
}

void coConfigEntry::setPath(const QString &path)
{
    this->path = path;
}

const QString &coConfigEntry::getPath() const
{
    return path;
}

const QString &coConfigEntry::getConfigName() const
{
    return configName;
}

QString coConfigEntry::getName() const
{

    if (this->name.isEmpty())
    {
        QString *nameAttrPtr = attributes["name"];
        if (!nameAttrPtr)
            nameAttrPtr = attributes["index"];
        if (nameAttrPtr)
        {
            QString nameAttr = *nameAttrPtr;
            cleanName(nameAttr);

            QString name = path.section('.', -1);
            cleanName(name);

            //cerr << name + "." + nameAttr << endl;
            this->name = name + ":" + nameAttr;
        }
        else
        {
            this->name = path.section('.', -1);
            cleanName(this->name);
        }
    }

    return this->name;
}

const char *coConfigEntry::getCName() const
{
    getName();
    if ((cName == 0) || (name != cName))
    {
        delete[] cName;
        cName = new char[name.length() + 20];
        strcpy(cName, name.toLatin1());
    }
    return cName;
}

coConfigEntryStringList coConfigEntry::getScopeList(QString scope)
{
    coConfigEntryStringList list;

    if (!matchingAttributes())
        return list;

    if (!scope.isNull())
    {

        QString childname;

        if (scope.contains('.'))
        {
            childname = scope.section('.', 0, 0);
            scope = scope.section('.', 1);
        }
        else
        {
            childname = scope;
            scope = QString::null;
        }

        for (coConfigEntryPtrList::iterator child = children.begin();
             child != children.end(); ++child)
        {
            if ((*child)->getName() == childname)
            {
                coConfigEntryStringList newList = (*child)->getScopeList(scope);
                list.merge(newList);
            }
        }

        return list;
    }

    if (isListNode)
    {
        COCONFIGDBG("coConfigEntry::getScopeList info: getting PLAIN_LIST");
        QStringList values;
        for (QStringList::iterator it = textNodes.begin(); it != textNodes.end(); ++it)
        {
            coConfigEntryString listEntry(*it, configScope, configName);
            list.append(listEntry);
        }
        list.setListType(coConfigEntryStringList::PLAIN_LIST);
    }
    else
    {
        COCONFIGDBG("coConfigEntry::getScopeList info: getting VARIABLE");
        for (coConfigEntryPtrList::iterator entry = children.begin();
             entry != children.end(); ++entry)
        {
            coConfigEntryString listEntry((*entry)->getName(), configScope, configName);
            list.append(listEntry);
        }
        list.setListType(coConfigEntryStringList::VARIABLE);
    }
    return list;
}

coConfigEntryStringList coConfigEntry::getVariableList(QString scope)
{
    coConfigEntryStringList list;
    list.setListType(coConfigEntryStringList::VARIABLE);

    if (!matchingAttributes())
        return list;

    if (!scope.isNull())
    {

        QString childname;

        if (scope.contains('.'))
        {
            childname = scope.section('.', 0, 0);
            scope = scope.section('.', 1);
        }
        else
        {
            childname = scope;
            scope = QString::null;
        }

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == childname)
                list += (*child)->getVariableList(scope);
        }

        return list;
    }

    QList<QString> keys = attributes.keys();

    for (QList<QString>::iterator key = keys.begin(); key != keys.end(); ++key)
    {
        coConfigEntryString listEntry((*key), configScope, configName);
        if (isListNode)
            listEntry.setListItem(true);
        list.append(listEntry);
    }

    return list;
}

coConfigEntryString coConfigEntry::getValue(const QString &variable, QString scope)
{
    if (!matchingAttributes())
    {
        return coConfigEntryString(QString::null);
    }

    if (!scope.isEmpty())
    {

        coConfigEntryString value(QString::null);
        QString childname;

        if (scope.contains('.'))
        {
            childname = scope.section('.', 0, 0);
            scope = scope.section('.', 1);
        }
        else
        {
            childname = scope;
            scope = QString();
        }

        for (coConfigEntryPtrList::iterator child = children.begin();
             child != children.end(); ++child)
        {
            if ((*child)->getName() == childname)
            {
                value = (*child)->getValue(variable, scope);
                if (!value.isNull())
                    break;
            }
        }

        return value;
    }

    QString var;
    if (!variable.isEmpty())
    {
        var = variable;
    }
    else
    {
        var = "value";
    }

    if (QString *attr = attributes[var])
    {
        //COCONFIGLOG("coConfigEntry::getValue info: variable " << var << " found");
        coConfigEntryString value = *attr;
        value.setConfigScope(configScope);
        value.setConfigName(configName);
        return value;
    }
    else
    {
        //COCONFIGLOG("coConfigEntry::getValue info: variable " << var << " not found");
        return coConfigEntryString(QString::null);
    }

    return coConfigEntryString(QString::null);
}

const char *coConfigEntry::getEntry(const char *variable)
{

    static size_t maxlen = 0;
    static char *entry = 0;

    if (!matchingAttributes())
    {
        return 0;
    }

    if (variable)
    {

        //COCONFIGLOG("coConfigEntry::getEntry info: variable " << variable);

        char *dotpos = (char *)strchr(variable, '.');

        if (dotpos)
        {

            const char *result = 0;

            size_t len = dotpos - variable + 1;

            if (len > maxlen)
            {
                delete[] entry;
                entry = new char[len];
                maxlen = len;
            }

            strncpy(entry, variable, len);
            entry[len - 1] = 0;

            for (coConfigEntryPtrList::iterator child = children.begin();
                 child != children.end() && result == 0; ++child)
            {
                if ((*child)->getName() == entry)
                {
                    result = (*child)->getEntry(&dotpos[1]);
                }
            }

            return result;
        }
        else
        {
            for (coConfigEntryPtrList::iterator child = children.begin();
                 child != children.end(); ++child)
            {
                if ((*child)->getName() == variable)
                {
                    return (*child)->getEntry(0);
                }
            }

            return 0;
        }
    }
    else
    {
        if (QString *attr = attributes["value"])
        {
            //COCONFIGLOG("coConfigEntry::getEntry info: value " << attr->latin1());
            return attr->toLatin1();
        }
        else
        {
            return 0;
        }
    }
}

bool coConfigEntry::setValue(const QString &variable,
                             const QString &value,
                             const QString &section)
{

    bool found = false;

    QString scope = section;
    if (!scope.isNull())
    {

        QString childname;

        if (scope.contains('.'))
        {
            childname = scope.section('.', 0, 0);
            scope = scope.section('.', 1);
        }
        else
        {
            childname = scope;
            scope = QString::null;
        }

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == childname)
                found |= (*child)->setValue(variable, value, scope);
        }

        return found;
    }

    if (isReadOnly())
        return false;

    QString var;
    if (!variable.isNull())
    {
        var = variable;
    }
    else
    {
        var = "value";
    }

    if (attributes[var])
    {
        attributes.remove(var);
    }

    attributes.insert(var, new QString(value));

    if (var == "name")
    {
        delete[] this->cName;
        this->cName = 0;
        this->name = QString::null;
    }
    else if (var == "index")
    {
        delete[] this->cName;
        this->cName = 0;
        this->name = QString::null;
    }
    entryChanged(); //msg Observer
    return true;

    //cerr << "coConfigEntry::setValue info: " << variable << " = " << value << endl;
}

void coConfigEntry::addValue(const QString &variable, const QString &value, const QString &section)
{

    QString childname;
    QString scope = section;

    coConfigEntryPtrList *listOne = new coConfigEntryPtrList();
    coConfigEntryPtrList *listTwo = new coConfigEntryPtrList();
    coConfigEntryPtrList *listSwap;

    // Filling list with this coConfigEntry
    listTwo->append(this);

    // Traverse the whole section
    while (!scope.isNull())
    {

        if (scope.contains('.'))
        {
            // Cut first part of the section
            childname = scope.section('.', 0, 0);
            scope = scope.section('.', 1);
        }
        else
        {
            childname = scope;
            scope = QString::null;
        }

        listOne->clear();

        // Look, if the section exists as a child of an entry in the list
        for (coConfigEntryPtrList::iterator entry = listTwo->begin();
             entry != listTwo->end(); ++entry)
        {
            for (coConfigEntryPtrList::iterator index = (*entry)->children.begin(); index != (*entry)->children.end(); ++index)
            {
                if ((*index)->getName() == childname)
                {
                    // As long as there are entries with the section name, add those to listOne
                    listOne->append(*index);
                }
            }
        }

        if (listOne->isEmpty())
        {
            // New entry, make section and possibly subsections
            if (scope.length() > 0)
                listTwo->first()->makeSection(childname + "." + scope);
            else
                listTwo->first()->makeSection(childname);
            break;
        }
        else
        {
            // Take the list of entries and use them as the base for the next search
            listSwap = listTwo;
            listTwo = listOne;
            listOne = listSwap;
        }
    }

    //   cerr << "coConfigEntry::addValue info: added " << variable << " = " << value << " in "
    //        << section << " (base: " << listTwo->first()->getName() << ")" << endl;
    setValue(variable, value, section);
}

void coConfigEntry::makeSection(const QString &section)
{

    if (section.isNull())
        return;

    if (isReadOnly())
    {
        COCONFIGLOG("coConfigEntry::makeSection fixme: modifying read only entry");
    }

    coConfigEntry *entry = new coConfigXercesEntry();

    QString variable;
    if (section.contains('.'))
    {
        variable = section.section('.', 0, 0);
    }
    else
    {
        variable = section;
    }

    if (variable.contains(':'))
    {
        entry->attributes.insert("name", new QString(variable.section(':', 1, 1)));
        variable = variable.section(':', 0, 0);
    }

    COCONFIGDBG("coConfigEntry::makeSection info: making section " << variable);

    entry->configName = configName;
    entry->configScope = configScope;

    entry->setPath(path + "." + variable);

    children.append(entry);

    if (section.contains('.'))
    {
        entry->makeSection(section.section('.', 1));
    }

    // append coConfigSchemaInfos for this new entry
    if (coConfigSchema::getInstance())
    {
        //entry->schemaInfos = coConfigSchema::getInstance()->getSchemaInfosForElement(qNodeName);
        //QString id = path + "." +
        entry->schemaInfos = coConfigSchema::getInstance()->getSchemaInfosForElement(entry->getPath());
        //COCONFIGDBG( "coConfigEntry::restoreFromDom info: entry " << qNodeName <<"  has SchemaInfos " << entry->getSchemaInfos());
    }
    else
        COCONFIGDBG("coConfigEntry::restoreFromDom err: no SchemaInfos yet");
}

bool coConfigEntry::deleteValue(const QString &variable, const QString &section)
{

    bool removed = false;

    QString scope = section;
    if (!scope.isNull())
    {

        QString childname;

        if (scope.contains('.'))
        {
            childname = scope.section('.', 0, 0);
            scope = scope.section('.', 1);
        }
        else
        {
            childname = scope;
            scope = QString::null;
        }

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {

            if ((*child)->getName() == childname)
            {
                bool removedSomething = (*child)->deleteValue(variable, scope);
                removed |= removedSomething;
                if (removedSomething && !(*child)->hasValues() && !(*child)->hasChildren())
                {
                    child = children.erase(child);
                }
            }
        }
        entryChanged(); //msg Observer
        return removed;
    }

    if (isReadOnly())
        return false;

    if (attributes[variable])
    {
        COCONFIGDBG("coConfigEntry::deleteValue info: deleting " << variable);
        attributes.remove(variable);
        entryChanged(); //msg Observer
        return true;
    }
    else if (textNodes.contains(variable))
    {
        COCONFIGDBG("coConfigEntry::deleteValue info: deleting " << variable);
        attributes.remove(variable);
        entryChanged(); //msg Observer
        return true;
    }
    else
    {
        return false;
    }
}

bool coConfigEntry::deleteSection(const QString &section)
{

    QString scope = section;
    bool removed = false;

    if (scope.contains('.'))
    {
        QString childname = scope.section('.', 0, 0);
        scope = scope.section('.', 1);

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == childname)
                removed |= (*child)->deleteSection(scope);
        }
    }
    else
    {

        // FIXME: If sub-entries are read only, they are deleted anyway
        if (isReadOnly())
            return false;

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == scope)
            {
                COCONFIGDBG("coConfigEntry::deleteSection info: removing " << (*child)->getPath());
                child = children.erase(child);
                removed = true;
            }
        }
    }
    entryChanged(); //msg Observer
    return removed;
}

bool coConfigEntry::hasValues() const
{
    return (!attributes.isEmpty() || !textNodes.isEmpty());
}

bool coConfigEntry::hasChildren() const
{
    return !children.isEmpty();
}

bool coConfigEntry::isList() const
{
    return isListNode;
}

QString &coConfigEntry::cleanName(QString &name)
{
    name.replace(QRegExp("\\."), "_");
    name.replace(':', "|");
    return name;
}

void coConfigEntry::setReadOnly(bool ro)
{
    readOnly = ro;
}

bool coConfigEntry::isReadOnly() const
{
    return readOnly;
}

coConfigSchemaInfos *coConfigEntry::getSchemaInfos()
{
    return schemaInfos;
}

void coConfigEntry::setSchemaInfos(coConfigSchemaInfos *infos)
{
    if (infos)
    {
        schemaInfos = infos;
    }
}
