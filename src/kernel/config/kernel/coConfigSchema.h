/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGSCHEMA_H
#define COCONFIGSCHEMA_H

#include <config/coConfigEntry.h>
#include <config/coConfigSchemaInfosList.h>
#include <QHash>
#include <QString>
#include <QStringList>

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
class coConfigSchemaInfosList;

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

    static void loadSchema(const QString &filename = 0);

    QStringList getGroupsFromSchema();
    coConfigSchemaInfosList *getSchemaInfosForGroup(const QString groupName);
    coConfigSchemaInfos *getSchemaInfosForElement(const QString &name);
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
    static QString fileName;

    coConfigSchemaInfos *createSchemaInfos(xercesc::XSElementDeclaration *elementDecl, const QString parent = QString(""));
    void sortInGroups();
    void walkTree();
    void addAnnotationsForElement(xercesc::XSElementDeclaration *elementDecl, coConfigSchemaInfos *schemaInfos);
    QHash<QString, QString> *createFromSchemaFileAnnotationsList(xercesc::XSAnnotationList *annoList);
    QHash<QString, QString> *createFromSchemaFileAnnotation(xercesc::XSAnnotation *anno);
    QString processSimpleTypeDefinition(xercesc::XSSimpleTypeDefinition *xsSimpleTypeDef);
    //QStringList processParticle(xercesc::XSParticle *xsParticle);
    QList<xercesc::XSElementDeclaration *> processParticle(xercesc::XSParticle *xsParticle);
    QString printCompositorTypeConnector(xercesc::XSModelGroup::COMPOSITOR_TYPE type);

    // key is path of element, value is pointer to the elements coConfigSchemaInfos
    QHash<QString, coConfigSchemaInfos *> elements;
    // key is name of group, value is list with all coConfigSchemaInfos
    QHash<QString, coConfigSchemaInfosList *> groups;
    bool walked;
};
}
#endif
