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
const QString &coConfigSchemaInfos::getElement()
{
    return element;
}

const QString &coConfigSchemaInfos::getElementPath()
{
    //returns sth like LOCAL.Cover
    return elementPath;
}

//shown name
const QString &coConfigSchemaInfos::getElementName()
{
    return elementName;
}

const QString &coConfigSchemaInfos::getElementDescription()
{
    return elementDescription;
}

const QString &coConfigSchemaInfos::getElementGroup()
{
    //fallback for elementGroup can be Path
    //if (elementGroup.isEmpty()) return elementPath;
    return elementGroup;
}

//NOTE
QStringList coConfigSchemaInfos::getElementAllowedChildren()
{
    return allowedChildren;
}

QList<QString> coConfigSchemaInfos::getAttributes()
{
    return attributes.keys();
}

void coConfigSchemaInfos::setReadableElementRule(const QString &rule)
{
    readableElementRule = rule;
}

void coConfigSchemaInfos::setElement(const QString &name)
{
    element = name;
}

void coConfigSchemaInfos::setElementPath(const QString &path)
{
    //path comes in as .COCONFIG... so cut this,
    elementPath = path.section(".", 2);
    //returns sth like LOCAL.Cover
}

void coConfigSchemaInfos::setElementName(const QString &name)
{
    elementName = name;
}

void coConfigSchemaInfos::setElementDescription(const QString &elDescription)
{
    elementDescription = elDescription;
}

void coConfigSchemaInfos::setElementGroup(const QString &group)
{
    elementGroup = group;
}

void coConfigSchemaInfos::setAllowedChildren(QStringList children)
{
    allowedChildren.clear();
    allowedChildren << children;
}

void coConfigSchemaInfos::addAttribute(const QString &attr, bool required, const QString &defValue,
                                       const QString &readableRule, const QString &regExpressionString, const QString &attrDescription)
{
    if (!attributes.contains(attr))
    {
        attrData data;
        data.required = required;
        data.defaultValue = defValue;
        data.readableRule = readableRule;
        data.attrDescription = attrDescription;
        data.regularExpressionString = regExpressionString;
        attributes.insert(attr, data);
    }
}

attrData *coConfigSchemaInfos::getAttributeData(const QString &attribute)
{
    if (attributes.contains(attribute))
    {
        return &attributes[attribute];
    }
    else
    {
        return 0;
    }
}
