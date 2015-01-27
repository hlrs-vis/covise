/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_LINKLISTHANDLER_H
#define ME_LINKLISTHANDLER_H

#include <QObject>
#include <QVector>

#include "nodes/MENodeLink.h"

class MENode;
class MEPort;
class MENodeLink;

class MELinkListHandler : public QObject
{
    Q_OBJECT

public:
    MELinkListHandler();
    ~MELinkListHandler();

    static MELinkListHandler *instance();

    void clearList();
    void resetLinks(MENode *node);
    void removeLinks(MENode *node);
    void addLink(MENode *out, MEPort *in, MENode *from, MEPort *to);
    void deleteLink(MENode *out, MEPort *in, MENode *from, MEPort *to);
    void updateInputDataPorts(MEPort *outputPort);
    void highlightPortAndLinks(MEPort *port, bool);
    MENodeLink *getLink(MENode *out, MEPort *in, MENode *from, MEPort *to);
    QVector<MENodeLink *> getLinksOut(MEPort *port);
    QVector<MENodeLink *> getLinksIn(MEPort *port);

private:
    QVector<MENodeLink *> linkList;
};
#endif
