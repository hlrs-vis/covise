/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGXERCESROOT_H
#define COCONFIGXERCESROOT_H

#include <QFile>
#include <QHash>

#include <config/coConfigRoot.h>
#include <config/coConfigEntry.h>
#include <config/coConfigEntryString.h>

#include <util/coTypes.h>

#include <xercesc/framework/psvi/XSAnnotation.hpp>
#include <xercesc/framework/psvi/XSNamedMap.hpp>

namespace XERCES_CPP_NAMESPACE
{
class DOMNode;
class SchemaGrammar;
class XercesDOMParser;
};

namespace covise
{

class coConfigGroup;

class coConfigXercesRoot : public coConfigRoot
{

public:
    coConfigXercesRoot(const QString &name, const QString &filename,
                       bool create = false, coConfigGroup *group = NULL);
    coConfigXercesRoot(const xercesc::DOMNode *node, const QString &name,
                       const QString &filename = QString::null, coConfigGroup *group = NULL);

    virtual ~coConfigXercesRoot();

    //      QHash<QString, QString>*  getSchemaInfosForNode (xercesc::DOMNode* node);

    virtual coConfigRoot *clone() const;
    virtual void merge(const coConfigRoot *with);

private:
    coConfigXercesRoot(const coConfigXercesRoot *source);

    void setContentsFromDom(const xercesc::DOMNode *node);

    xercesc::DOMNode *loadFile(const QString &filename);

    virtual void load(bool create = false);

    virtual void createGlobalConfig();
    virtual void createClusterConfig(const QString &hostname);
    virtual void createHostConfig(const QString &hostname);
};
}
#endif
