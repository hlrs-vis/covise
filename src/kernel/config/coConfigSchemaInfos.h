/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONFIGSCHEMAINFOS
#define CONFIGSCHEMAINFOS

// #include <config/coConfigSchema.h>
#include <map>
#include <set>
#include <string>
#include <util/coTypes.h>
#include <vector>

// holds the data for one attribute

namespace covise
{

typedef struct
{
    bool required;
    std::string defaultValue, readableRule, regularExpressionString, attrDescription;
} attrData;

class CONFIGEXPORT coConfigSchemaInfos
{
    friend class coConfigSchema; // needs all the setter methods
public:
    coConfigSchemaInfos();
    ~coConfigSchemaInfos();

    const std::string &getElement(); // real name
    const std::string &getElementPath();
    const std::string &getElementName(); // shown name
    const std::string &getElementDescription();
    const std::string &getElementGroup(); // group to that the element belongs
    const std::set<std::string> &getElementAllowedChildren() const;
    std::set<std::string> getAttributes() const; // list of all attributes for this element
    // pointer to the attribute data
    attrData *getAttributeData(const std::string &attribute);

private:
    void setElement(const std::string &name);
    void setReadableElementRule(const std::string &rule);
    void setElementPath(const std::string &path);
    void setElementName(const std::string &name);
    void setElementDescription(const std::string &elDescription);
    void setElementGroup(const std::string &group);
    void setAllowedChildren(const std::set<std::string> &children);
    void addAttribute(const std::string &attr, bool required, const std::string &defValue, const std::string &readableRule = "",
                      const std::string &regExpressionString = "", const std::string &attrDescription = "");

    std::string element, elementPath, readableElementRule, elementName, elementDescription, elementGroup;
    std::set<std::string> allowedChildren;
    std::map<std::string, attrData> attributes;
};

typedef std::set<coConfigSchemaInfos *> coConfigSchemaInfosList;
}
#endif
