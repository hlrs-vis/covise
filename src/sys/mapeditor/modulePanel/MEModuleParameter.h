/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MODULEPARAMETER_H
#define ME_MODULEPARAMETER_H

#include <QScrollArea>

class QGridLayout;
class QResizeEvent;
class QScrollArea;

class MENode;

//================================================
class MEModuleParameter : public QScrollArea
//================================================
{
    Q_OBJECT

public:
    MEModuleParameter(QWidget *parent, MENode *);
    ~MEModuleParameter();

    MENode *getNode()
    {
        return m_node;
    };
    void paramChanged(bool);

signals:

    void disableDiscard(bool);

private:
    MENode *m_node;
    QWidget *m_main;

    int getLocalType(int);
};
#endif
