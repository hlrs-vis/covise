/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigSchemaInfos.h>

using namespace covise;

coConfigSchemaInfos::coConfigSchemaInfos()
{
    elementGroup = "";
}

coConfigSchemaInfos::~coConfigSchemaInfos()
{
}

//real name
const std::string &coConfigSchemaInfos::getElement()
{
    return element;
}

const std::string &coConfigSchemaInfos::getElementPath()
{
    //returns sth like LOCAL.Cover
    return elementPath;
}

//shown name
const std::string &coConfigSchemaInfos::getElementName()
{
    return elementName;
}

const std::string &coConfigSchemaInfos::getElementDescription()
{
    return elementDescription;
}

const std::string &coConfigSchemaInfos::getElementGroup()
{
    //fallback for elementGroup can be Path
    //if (elementGroup.isEmpty()) return elementPath;
    return elementGroup;
}

//NOTE
const std::set<std::string> &coConfigSchemaInfos::getElementAllowedChildren() const
{
    return allowedChildren;
}

std::set<std::string> coConfigSchemaInfos::getAttributes() const
{
    std::set<std::string> s;
    for (const auto &attribute : attributes)
        s.insert(attribute.first);
    return s;
}

void coConfigSchemaInfos::setReadableElementRule(const std::string &rule)
{
    readableElementRule = rule;
}

void coConfigSchemaInfos::setElement(const std::string &name)
{
    element = name;
}

void coConfigSchemaInfos::setElementPath(const std::string &path)
{
    //path comes in as .COCONFIG... so cut this,
    size_t num = 0;
    for (size_t i = 0; i < path.size(); i++)
    {
        if (path[i] == '.')
            ++num;
        if (num == 2)
        {
            elementPath = path.substr(++i);
            break;
        }
    }
    // returns sth like LOCAL.Cover
}

void coConfigSchemaInfos::setElementName(const std::string &name)
{
    elementName = name;
}

void coConfigSchemaInfos::setElementDescription(const std::string &elDescription)
{
    elementDescription = elDescription;
}

void coConfigSchemaInfos::setElementGroup(const std::string &group)
{
    elementGroup = group;
}

void coConfigSchemaInfos::setAllowedChildren(const std::set<std::string> &children)
{
    allowedChildren = children;
}

void coConfigSchemaInfos::addAttribute(const std::string &attr, bool required, const std::string &defValue,
                                       const std::string &readableRule, const std::string &regExpressionString, const std::string &attrDescription)
{
    if (attributes.find(attr) == attributes.end())
    {
        attrData data;
        data.required = required;
        data.defaultValue = defValue;
        data.readableRule = readableRule;
        data.attrDescription = attrDescription;
        data.regularExpressionString = regExpressionString;
        attributes.insert({attr, data});
    }
}

attrData *coConfigSchemaInfos::getAttributeData(const std::string &attribute)
{
    auto it = attributes.find(attribute);
    if(it != attributes.end())
        return &it->second;
    else
        return nullptr;
}
