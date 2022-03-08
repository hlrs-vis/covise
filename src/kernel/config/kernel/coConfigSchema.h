/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGSCHEMA_H
#define COCONFIGSCHEMA_H

#include <config/coConfigEntry.h>
#include <string>
#include <set>
#include <map>

#include <xercesc/framework/psvi/XSConstants.hpp>
#include <xercesc/framework/psvi/XSModelGroup.hpp>
#include <xercesc/framework/psvi/XSNamedMap.hpp>

namespace XERCES_CPP_NAMESPACE
{
class XSModel;
class XSSimpleTypeDefinition;
class XSAnnotation;
class XSElementDeclaration;
};
namespace covise
{

class coConfigSchemaInfos;
typedef std::set<coConfigSchemaInfos *> coConfigSchemaInfosList;

class CONFIGEXPORT coConfigSchema
{
public:
    static coConfigSchema *getInstance(xercesc::XSModel *model = 0)
    {
        if (configSchema)
        {
            return configSchema;
        }
        if (model)
        {
            configSchema = new coConfigSchema(model);
            configSchema->walkTree();
            return configSchema;
        }
        return 0;
    }

    static void loadSchema(const std::string &filename = 0);

    std::set<std::string> getGroupsFromSchema();
    coConfigSchemaInfosList *getSchemaInfosForGroup(const std::string groupName);
    coConfigSchemaInfos *getSchemaInfosForElement(const std::string &name);
    //    coConfigSchemaInfos* getSchemaInfos(xercesc::XSElementDeclaration* elementDecl);

protected:
    coConfigSchema(xercesc::XSModel *model)
    {
        xsModel = model;
        walked = 0;
    }
    ~coConfigSchema();

private:
    static xercesc::XSModel *xsModel;
    static coConfigSchema *configSchema;
    static std::string fileName;

    coConfigSchemaInfos *createSchemaInfos(xercesc::XSElementDeclaration *elementDecl, const std::string parent = std::string(""));
    void sortInGroups();
    void walkTree();
    void addAnnotationsForElement(xercesc::XSElementDeclaration *elementDecl, coConfigSchemaInfos *schemaInfos);
    std::map<std::string, std::string> createFromSchemaFileAnnotationsList(xercesc::XSAnnotationList *annoList);
    std::map<std::string, std::string> createFromSchemaFileAnnotation(xercesc::XSAnnotation *anno);
    std::string processSimpleTypeDefinition(xercesc::XSSimpleTypeDefinition *xsSimpleTypeDef);
    // std::set<std::string> processParticle(xercesc::XSParticle *xsParticle);
    std::vector<xercesc::XSElementDeclaration *> processParticle(xercesc::XSParticle *xsParticle);
    std::string printCompositorTypeConnector(xercesc::XSModelGroup::COMPOSITOR_TYPE type);

    // key is path of element, value is pointer to the elements coConfigSchemaInfos
    std::map<std::string, coConfigSchemaInfos *> elements;
    // key is name of group, value is list with all coConfigSchemaInfos
    std::map<std::string, coConfigSchemaInfosList> groups;
    bool walked;
};
}
#endif
