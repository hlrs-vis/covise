/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <list>

#include <xercesc/dom/DOM.hpp>

#include <config/coConfigLog.h>
#include <config/coConfigSchemaInfos.h>
#include "coConfigSchema.h"
#include "coConfigXercesEntry.h"
#include "coConfigTools.h"
#include "coConfigXercesConverter.h"

#include <util/string_util.h>
using namespace covise;

// #include <config/coConfigEvent.h>

coConfigXercesEntry::coConfigXercesEntry()
{
}

coConfigXercesEntry::~coConfigXercesEntry()
{
}

coConfigEntry::coConfigEntry(const coConfigEntry *source)
    : configScope(source->configScope), configName(source->configName), path(source->path), isListNode(source->isListNode), readOnly(source->readOnly), textNodes(source->textNodes), schemaInfos(source->schemaInfos), elementGroup(source->elementGroup), cName(0), name(source->name), attributes(source->attributes)
{
    for (coConfigEntryPtrList::const_iterator child = source->children.begin();
         child != source->children.end(); ++child)
    {
        coConfigEntry *childClone = (*child)->clone();
        if (childClone)
            this->children.emplace_back(childClone);
        else
            COCONFIGDBG("coConfigEntry::<init> err: child clone returned 0");
    }
}

coConfigXercesEntry::coConfigXercesEntry(const coConfigXercesEntry *source)
    : coConfigEntry(source)
{
}

void coConfigEntry::entryChanged()
{
    //COCONFIGDBG("coConfigEntry::entryChanged(): info: notify called  " << this->getName());
    notify();
}

const coConfigEntryPtrList &coConfigEntry::getChildren() const
{
    return children;
}

bool coConfigEntry::matchingAttributes() const
{
    return coConfigTools::matchingAttributes(attributes);
}

xercesc::DOMNode *coConfigXercesEntry::storeToDom(xercesc::DOMDocument &document, int indent)
{
    xercesc::DOMElement *element = document.createElement(stringToXexcesc(split(path, '.').back()).get());

    if (xercesc::DOMNode::ELEMENT_NODE == element->getNodeType())
    {
        bool include = xercescToStdString(element->getNodeName()) == "INCLUDE";
        bool tryinclude = xercescToStdString(element->getNodeName()) == "TRYINCLUDE";
        if (include || tryinclude)
        {
            if (isListNode)
            {
                for (const auto &attribute : attributes)
                {
                    element->setAttribute(stringToXexcesc(attribute.first).get(), stringToXexcesc(attribute.second).get());
                    COCONFIGDBG("coConfigEntry::storeToDom info: includenode");
                }
                element->appendChild(document.createTextNode(xercesc::XMLString::transcode("\n  ")));
                std::string values;
                for (const auto &textNode : textNodes)
                {
                    values.append(textNode + "\n ");
                }
                values.pop_back();
                values.pop_back();
                element->appendChild(document.createTextNode(stringToXexcesc(values).get()));
            }
            element->appendChild(document.createTextNode(xercesc::XMLString::transcode("\n")));
        }
        else
        {
            for (const auto &attribute : attributes)
            {
                element->setAttribute(stringToXexcesc(attribute.first).get(), stringToXexcesc(attribute.second).get());
                COCONFIGDBG("coConfigEntry::storeToDom info: noIncludenode key " << attribute.first << " attr " << attribute.second);
            }

            if (isListNode && !textNodes.empty())
            {
                std::string values = std::string(indent, ' ') + *textNodes.begin();
                std::string ending = "\n" + std::string(indent, ' ');

                for (auto i = ++textNodes.begin(); i != textNodes.end(); i++)
                {
                    values += ending + *i;
                }
                element->appendChild(document.createTextNode(stringToXexcesc(values).get()));
            }

            for (coConfigEntryPtrList::iterator entry = children.begin();
                 entry != children.end(); ++entry)
            {
                COCONFIGDBG("coConfigEntry::storeToDom info: children " << (*entry)->getName());
                element->appendChild(document.createTextNode(stringToXexcesc("\n" + std::string(indent, ' ')).get()));
                coConfigXercesEntry *xentry = dynamic_cast<coConfigXercesEntry *>(entry->get());
                if (xentry)
                    element->appendChild((xentry)->storeToDom(document, indent + 2));
                COCONFIGDBG("coConfigEntry::storeToDom info: childexit " << (*entry)->getName());
            }

            if (indent > 2)
                element->appendChild(document.createTextNode(stringToXexcesc("\n" + std::string(indent - 2, ' ')).get()));
            else
                element->appendChild(document.createTextNode(xercesc::XMLString::transcode("\n")));
        }
    }

    return element;
}

coConfigEntry *coConfigXercesEntry::restoreFromDom(xercesc::DOMElement *node,
                                                   const std::string &configName)
{
    coConfigEntry *entry = new coConfigXercesEntry();
    std::string qNodeName = xercescToStdString(node->getNodeName());
    std::string path = qNodeName;
    entry->configName = configName;

    if (configName.empty())
    {
        COCONFIGLOG("coConfigEntry::restoreFromDom warn: no config name: ");
    }

    // Tree upward iteration to get scope attribute and complete path ,if not already one step before yac
    xercesc::DOMNode *scopeNode = node;
    if (node->getParentNode() == 0 && toUpper(xercescToStdString(node->getNodeName())) != "COCONFIG" && node->getNodeType() != xercesc::DOMNode::DOCUMENT_NODE)
    {
    }
    else
    { //iterate parents
        for (xercesc::DOMNode *parent = node->getParentNode();
             parent != 0 &&
             toUpper(xercescToStdString(parent->getNodeName())) != "COCONFIG" && parent->getNodeType() != xercesc::DOMNode::DOCUMENT_NODE;
             parent = parent->getParentNode())
        {
            path = xercescToStdString(parent->getNodeName()) + "." + path;
            scopeNode = parent;
        }
    }

    XMLCh *scopeTag = xercesc::XMLString::transcode("scope");

    if (xercesc::DOMNode::ELEMENT_NODE == scopeNode->getNodeType())
    {
        xercesc::DOMElement *scopeElement = static_cast<xercesc::DOMElement *>(scopeNode);
        std::string configScope = xercescToStdString(scopeElement->getAttribute(scopeTag));
        if (toUpper(xercescToStdString(scopeElement->getNodeName())) == "GLOBAL")
        {
            entry->configScope = coConfigConstants::Global;
        }
        else if (toUpper(xercescToStdString(scopeElement->getNodeName())) == "LOCAL")
        {
            entry->configScope = coConfigConstants::Host;
        }
        else if (toUpper(xercescToStdString(scopeElement->getNodeName())) == "CLUSTER")
        {
            entry->configScope = coConfigConstants::Cluster;
        }
        else if (!configScope.empty())
        {
            COCONFIGLOG("coConfigEntry::restoreFromDom warn: unknown config scope <" << configScope << ">");
        }
    }
    xercesc::XMLString::release(&scopeTag);
    entry->setPath(path);

    // get node attributes and their values
    xercesc::DOMNamedNodeMap *map = node->getAttributes();
    for (int ctr = 0; ctr < map->getLength(); ++ctr)
        entry->attributes.insert({xercescToStdString(map->item(ctr)->getNodeName()), xercescToStdString(map->item(ctr)->getNodeValue())});

    // collect all child nodes. if text, add. if element start recursition
    xercesc::DOMNodeList *nodeList = node->getChildNodes();
    for (int i = 0; i < nodeList->getLength(); ++i)
    {
        //if textNode, not empty line, it is added to entry
        if (nodeList->item(i)->getNodeType() == xercesc::DOMNode::TEXT_NODE)
        {
            std::string nodeValue = xercescToStdString(nodeList->item(i)->getNodeValue());
            if (!strip(nodeValue).empty())
            {
                entry->isListNode = true;
                auto textlist = split(nodeValue, '\n', true);
                for (auto text : textlist)
                {
                    std::string singleNodeValue = strip(text);
                    if (!singleNodeValue.empty())
                    {
                        entry->attributes[text] = "";
                        entry->textNodes.insert(text);
                    }
                }
            }
        }
        if (xercesc::DOMNode::ELEMENT_NODE == nodeList->item(i)->getNodeType())
        {
            if (coConfigXercesEntry *xentry = dynamic_cast<coConfigXercesEntry *>(entry))
                xentry->children.emplace_back(coConfigXercesEntry::restoreFromDom(static_cast<xercesc::DOMElement *>(nodeList->item(i)), configName));
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
                (*myEntry)->merge((entry->get()));
                break;
            }
        }

        if (myEntry == children.end())
            children.emplace_back((*entry)->clone());
    }

    attributes.insert(with->attributes.begin(), with->attributes.end());
}

coConfigEntry *coConfigXercesEntry::clone() const
{
    return new coConfigXercesEntry(this);
}

void coConfigEntry::setPath(const std::string &path)
{
    this->path = path;
}

const std::string &coConfigEntry::getPath() const
{
    return path;
}

const std::string &coConfigEntry::getConfigName() const
{
    return configName;
}

std::string coConfigEntry::getName() const
{
    if (name.empty())
    {
        name = split(path, '.').back();
        cleanName(name);

        std::string attr;
        auto nameAttrPtr = std::find_if(attributes.begin(), attributes.end(), [](const std::pair<std::string, std::string> &p)
                                        { return p.first == "name" || p.first == "index"; });
        if (nameAttrPtr != attributes.end())
        {
            std::string nameAttr = nameAttrPtr->second;
            cleanName(nameAttr);

            name = split(path, '.').back();
            cleanName(name);
            name += ":" + nameAttr;
        }
    }

    return name;
}

const char *coConfigEntry::getCName() const
{
    getName();
    return name.c_str();
}

struct Scope
{
    Scope(const std::string &scope)
    {
        if (!scope.empty())
        {
            auto pos = scope.find('.');
            if (pos != std::string::npos)
            {
                childName = scope.substr(0, pos);
                this->scope = scope.substr(pos + 1);
            }
            else
            {
                childName = scope;
            }
        }
    }

    std::string scope, childName;
};

coConfigEntryStringList
coConfigEntry::getScopeList(const std::string &scope)
{
    coConfigEntryStringList list;

    if (!matchingAttributes())
        return list;

    if (!scope.empty())
    {

        Scope s(scope);

        for (coConfigEntryPtrList::iterator child = children.begin();
             child != children.end(); ++child)
        {
            if ((*child)->getName() == s.childName)
            {
                coConfigEntryStringList newList = (*child)->getScopeList(s.scope);
                list.merge(newList);
            }
        }

        return list;
    }

    if (isListNode)
    {
        COCONFIGDBG("coConfigEntry::getScopeList info: getting PLAIN_LIST");
        list.entries().insert(textNodes.begin(), textNodes.end());
        list.setListType(coConfigEntryStringList::PLAIN_LIST);
    }
    else
    {
        COCONFIGDBG("coConfigEntry::getScopeList info: getting VARIABLE");
        for (const auto &entry : children)
        {
            coConfigEntryString listEntry{entry->getName(), configName, "", configScope};
            list.entries().insert(listEntry);
        }
        list.setListType(coConfigEntryStringList::VARIABLE);
    }
    return list;
}

coConfigEntryStringList coConfigEntry::getVariableList(const std::string &scope)
{
    coConfigEntryStringList list;
    appendVariableList(list, scope);
    return list;
}

void coConfigEntry::appendVariableList(coConfigEntryStringList &list, const std::string &scope)
{

    list.setListType(coConfigEntryStringList::VARIABLE);

    if (matchingAttributes())
    {
        if (!scope.empty())
        {
            Scope s(scope);
            for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
            {
                if ((*child)->getName() == s.childName)
                    (*child)->appendVariableList(list, s.scope);
            }
        }
        else
        {
            for (const auto &attribute : attributes)
            {
                coConfigEntryString listEntry{attribute.first, configName, "", configScope};
                if (isListNode)
                    listEntry.islistItem = true;
                list.entries().insert(listEntry);
            }
        }
    }
}

coConfigEntryString coConfigEntry::getValue(const std::string &variable, const std::string &scope)
{
    if (!matchingAttributes())
    {
        return coConfigEntryString();
    }

    if (!scope.empty())
    {

        Scope s(scope);
        coConfigEntryString value;

        for (coConfigEntryPtrList::iterator child = children.begin();
             child != children.end(); ++child)
        {
            if ((*child)->getName() == s.childName)
            {
                value = (*child)->getValue(variable, s.scope);
                if (!(value == coConfigEntryString{}))
                    break;
            }
        }

        return value;
    }

    std::string var;
    if (!variable.empty())
        var = variable;
    else
        var = "value";

    auto attr = attributes.find(var);
    if (attr != attributes.end())
    {
        // COCONFIGLOG("coConfigEntry::getValue info: variable " << var << " found");
        coConfigEntryString value{attr->second, configName, "", configScope};
        return value;
    }
    else
    {
        // COCONFIGLOG("coConfigEntry::getValue info: variable " << var << " not found");
        return coConfigEntryString();
    }

    return coConfigEntryString();
}

const char *coConfigEntry::getEntry(const char *variable)
{

    static size_t maxlen = 0;
    static char *entry = 0;

    if (!matchingAttributes())
        return 0;

    if (variable)
    {

        // COCONFIGLOG("coConfigEntry::getEntry info: variable " << variable);

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
        auto attr = attributes.find("value");
        if (attr != attributes.end() && !attr->second.empty())
        {
            // COCONFIGLOG("coConfigEntry::getEntry info: value " << attr->latin1());
            return attr->second.c_str();
        }
        else
        {
            return 0;
        }
    }
}

bool coConfigEntry::setValue(const std::string &variable,
                             const std::string &value,
                             const std::string &section)
{

    bool found = false;

    if (!section.empty())
    {

        Scope s(section);

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == s.childName)
                found |= (*child)->setValue(variable, value, s.scope);
        }

        return found;
    }

    if (isReadOnly())
        return false;

    std::string var;
    if (!variable.empty())
        var = variable;
    else
        var = "value";

    attributes[var] = value;

    if (var == "name")
    {
        delete[] this->cName;
        this->cName = 0;
        this->name = std::string();
    }
    else if (var == "index")
    {
        delete[] this->cName;
        this->cName = 0;
        this->name = std::string();
    }
    entryChanged(); // msg Observer
    return true;

    // cerr << "coConfigEntry::setValue info: " << variable << " = " << value << endl;
}

void coConfigEntry::addValue(const std::string &variable, const std::string &value, const std::string &section)
{

    std::vector<coConfigEntry *> listOne, listTwo;

    // Filling list with this coConfigEntry
    listTwo.push_back(this);

    // Traverse the whole section
    Scope s(section);

    while (!s.scope.empty())
    {

        s = Scope(s.scope);

        listOne.clear();

        // Look, if the section exists as a child of an entry in the list
        for (auto entry = listTwo.begin();
             entry != listTwo.end(); ++entry)
        {
            for (coConfigEntryPtrList::iterator index = (*entry)->children.begin(); index != (*entry)->children.end(); ++index)
            {
                if ((*index)->getName() == s.childName)
                {
                    // As long as there are entries with the section name, add those to listOne
                    listOne.push_back(index->get());
                }
            }
        }

        if (listOne.size() == 0)
        {
            // New entry, make section and possibly subsections
            if (s.scope.length() > 0)
                (*(listTwo.begin()))->makeSection(s.childName + "." + s.scope);
            else
                (*(listTwo.begin()))->makeSection(s.childName);
            break;
        }
        else
        {
            // Take the list of entries and use them as the base for the next search
            std::swap(listOne, listTwo);
        }
    }

    //   cerr << "coConfigEntry::addValue info: added " << variable << " = " << value << " in "
    //        << section << " (base: " << listTwo->first()->getName() << ")" << endl;
    setValue(variable, value, section);
}

void coConfigEntry::makeSection(const std::string &section)
{

    if (section.empty())
        return;

    if (isReadOnly())
    {
        COCONFIGLOG("coConfigEntry::makeSection fixme: modifying read only entry");
    }

    coConfigEntry *entry = new coConfigXercesEntry();
    Scope s(section);
    auto sp = split(s.childName, ':');
    if (sp.size() > 1)
    {
        entry->attributes.insert({"name", sp[1]});
        s.childName = sp[0];
    }

    COCONFIGDBG("coConfigEntry::makeSection info: making section " << s.childName);

    entry->configName = configName;
    entry->configScope = configScope;

    entry->setPath(path + "." + s.childName);

    children.emplace_back(entry);

    if (!s.scope.empty())
    {
        entry->makeSection(s.scope);
    }

    // append coConfigSchemaInfos for this new entry
    if (coConfigSchema::getInstance())
    {
        // entry->schemaInfos = coConfigSchema::getInstance()->getSchemaInfosForElement(qNodeName);
        // std::string id = path + "." +
        entry->schemaInfos = coConfigSchema::getInstance()->getSchemaInfosForElement(entry->getPath());
        //COCONFIGDBG( "coConfigEntry::restoreFromDom info: entry " << qNodeName <<"  has SchemaInfos " << entry->getSchemaInfos());
    }
    else
        COCONFIGDBG("coConfigEntry::restoreFromDom err: no SchemaInfos yet");
}

bool coConfigEntry::deleteValue(const std::string &variable, const std::string &section)
{

    bool removed = false;

    if (!section.empty())
    {

        Scope s(section);

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {

            if ((*child)->getName() == s.childName)
            {
                bool removedSomething = (*child)->deleteValue(variable, s.scope);
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

    auto attribute = attributes.find(variable);
    if (attribute != attributes.end())
    {
        COCONFIGDBG("coConfigEntry::deleteValue info: deleting " << variable);
        attributes.erase(attribute);
        entryChanged(); //msg Observer
        return true;
    }
    else if (textNodes.find(variable) != textNodes.end())
    {
        COCONFIGDBG("coConfigEntry::deleteValue info: deleting " << variable);
        attributes.erase(variable);
        entryChanged(); //msg Observer
        return true;
    }
    else
    {
        return false;
    }
}

bool coConfigEntry::deleteSection(const std::string &section)
{

    bool removed = false;
    Scope s(section);
    if (s.scope.empty())
    {
        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == s.childName)
                removed |= (*child)->deleteSection(s.scope);
        }
    }
    else
    {

        // FIXME: If sub-entries are read only, they are deleted anyway
        if (isReadOnly())
            return false;

        for (coConfigEntryPtrList::iterator child = children.begin(); child != children.end(); ++child)
        {
            if ((*child)->getName() == section)
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
    return (!attributes.empty() || !textNodes.empty());
}

bool coConfigEntry::hasChildren() const
{
    return children.size()>0;
}

bool coConfigEntry::isList() const
{
    return isListNode;
}

std::string &coConfigEntry::cleanName(std::string &name)
{
    name = replace(name, ".", "_", -1);
    name = replace(name, ":", "|", -1);
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
