/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_INTSCALARPORT_H
#define ME_INTSCALARPORT_H

#include "ports/MEScalarPort.h"

//================================================
class MEIntScalarPort : public MEScalarPort
//================================================
{

    Q_OBJECT

public:
    MEIntScalarPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEIntScalarPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEIntScalarPort();

    void makeLayout(layoutType type, QWidget *parent);

#ifdef YAC
    void setValues(covise::coRecvBuffer &rb);
#endif

private slots:

    void textCB(const QString &);
    void text2CB(const QString &);
    void boundaryCB();

private:
    void plusNewValue();
    void minusNewValue();
};
#endif
