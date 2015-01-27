/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONFIGSCHEMAINFOS
#define CONFIGSCHEMAINFOS

// #include <config/coConfigSchema.h>
#include <util/coTypes.h>
#include <QString>
#include <QRegExp>
#include <QHash>
#include <QList>
#include <QStringList>

// holds the data for one attribute

namespace covise
{

typedef struct
{
    bool required;
    QString defaultValue, readableRule, regularExpressionString, attrDescription;
} attrData;

class CONFIGEXPORT coConfigSchemaInfos
{
    friend class coConfigSchema; // needs all the setter methods
public:
    coConfigSchemaInfos();
    ~coConfigSchemaInfos();

    const QString &getElement(); // real name
    const QString &getElementPath();
    const QString &getElementName(); // shown name
    const QString &getElementDescription();
    const QString &getElementGroup(); // group to that the element belongs
    QStringList getElementAllowedChildren();
    QList<QString> getAttributes(); // list of all attributes for this element
    // pointer to the attribute data
    attrData *getAttributeData(const QString &attribute);

private:
    void setElement(const QString &name);
    void setReadableElementRule(const QString &rule);
    void setElementPath(const QString &path);
    void setElementName(const QString &name);
    void setElementDescription(const QString &elDescription);
    void setElementGroup(const QString &group);
    void setAllowedChildren(QStringList children);
    void addAttribute(const QString &attr, bool required, const QString &defValue, const QString &readableRule = 0,
                      const QString &regExpressionString = 0, const QString &attrDescription = 0);

    QString element, elementPath, readableElementRule, elementName, elementDescription, elementGroup;
    QStringList allowedChildren;
    QHash<QString, attrData> attributes;
};
}
#endif
