/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MIRRORLISTHANDLER_H
#define ME_MIRRORLISTHANDLER_H

#include <QObject>

class cyNode;

class cyMirrorListHandler : public QObject
{
    Q_OBJECT

public:
    cyMirrorListHandler();
    ~cyMirrorListHandler();

    static cyMirrorListHandler *instance();

    void updateMirrorBox();
    void mirrorNodes();
    void setMirror();
    void addMirrorNodes();
    void delMirrorNodes();
    void selectMirrorHost(int);
    void mirrorStateChanged(int);

private:
    QVector<cyNode *> mirrorList; // node list contains synced nodes
};

#endif
