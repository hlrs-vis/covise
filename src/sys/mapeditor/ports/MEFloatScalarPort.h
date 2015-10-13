/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_SCALARPORT_H
#define ME_SCALARPORT_H

#include "ports/MEScalarPort.h"

//================================================
class MEFloatScalarPort : public MEScalarPort
//================================================
{

    Q_OBJECT

public:
    MEFloatScalarPort(MENode *node, QGraphicsScene *scene, const QString &portname, const QString &paramtype, const QString &description);
    MEFloatScalarPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEFloatScalarPort();

    void makeLayout(layoutType type, QWidget *parent);

private slots:

    void textCB(const QString &value);
    void text2CB(const QString &value);
    void boundaryCB();

private:
    void plusNewValue();
    void minusNewValue();
};
#endif
