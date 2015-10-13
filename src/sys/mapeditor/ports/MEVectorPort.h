/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_VECTORPORT_H
#define ME_VECTORPORT_H

#include <QVector>

#include "ports/MEParameterPort.h"

class QVariant;
class MELineEdit;

//================================================
class MEVectorPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEVectorPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEVectorPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEVectorPort();

    void restoreParam();
    void storeParam();
    void moduleParameterRequest();
    void defineParam(QString value, int apptype);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void modifyParameter(QString value);
    void makeLayout(layoutType type, QWidget *parent);

protected:
    int m_nVect;
    QVector<QVariant> m_vector, m_vectorold;
    QVector<MELineEdit *> m_vectorList[2];

    void removeFromControlPanel();
    void sendParamMessage();

private:
    QString asString(int idx) const;
};
#endif
