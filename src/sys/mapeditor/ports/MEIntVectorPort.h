/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_INTVECTORPORT_H
#define ME_INTVECTORPORT_H

#include "ports/MEVectorPort.h"

//================================================
class MEIntVectorPort : public MEVectorPort
//================================================
{

    Q_OBJECT

public:
    MEIntVectorPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEIntVectorPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEIntVectorPort();

#ifdef YAC
    void setValues(covise::coRecvBuffer &);
#endif

private slots:

    void textCB(const QString &);
};
#endif
