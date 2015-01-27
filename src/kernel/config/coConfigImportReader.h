/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGIMPORTREADER_H
#define COCONFIGIMPORTREADER_H

#include <util/coTypes.h>

#include <QLinkedList>
#include <QMap>
#include <QTextStream>
#include <QtXml>

class QDomDocument;
class QDomElement;
class QDomNode;
class QFile;
class QString;
class QTextStream;

#ifndef CO_gcc3
EXPORT_TEMPLATE2(template class CONFIGEXPORT QMap<QString, QDomElement>)
#endif

class CONFIGEXPORT coConfigImportReader
{

public:
    coConfigImportReader(const QString &source,
                         const QString &dest,
                         const QString &transform,
                         bool resolve_includes = true);

    ~coConfigImportReader();

    QDomDocument parse();
    QDomDocument write();

    void updatev0v1();

private:
    QDomElement findMapping(const QString &mapping, const QDomNode &parent) const;

    QDomElement getOrCreateSection(QDomDocument &document, QDomNode &parent, const QString &name, const QString &hosts = "");
    QDomNode addChild(QDomNode &parent, const QDomNode &newChild);

    void update(QDomElement &rootNode, QDomElement &updater);
    inline void updateNode(QDomElement &node, QDomElement &updateInstruction);
    inline void updateEntry(QDomElement &node, QDomElement &updateInstruction);
    inline void updateApplyInstruction(QDomElement &node, QDomElement &updateInstruction);
    inline void updateMergeNodes(QDomElement &section, QDomElement &node);
    inline QLinkedList<QDomNode> makeNonLiveList(QDomNodeList liveList) const;

    inline QString domToString(const QDomDocument &doc) const;

    QFile *source;
    QFile *dest;
    QFile *transform;

    QTextStream *sourceStream;
    QDomDocument transformerDoc;
    QDomElement transformer;
    QDomDocument result;
    QDomNode root;
    QDomNode global;
    QString errorMessage;

    QMap<QString, QDomElement> hostconfigs;

    int errorLine;
    int errorColumn;
    bool resolveIncludes;
};

#endif
