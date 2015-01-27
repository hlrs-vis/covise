/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FLOATSCALARPORT_H
#define ME_FLOATSCALARPORT_H

#include "ports/MEParameterPort.h"

class QVariant;
class MELineEdit;

//================================================
class MEScalarPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEScalarPort(MENode *node, QGraphicsScene *scene, const QString &portname, const QString &paramtype, const QString &description);
    MEScalarPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEScalarPort();

    void restoreParam();
    void storeParam();
    void moduleParameterRequest();
    void defineParam(QString value, int apptype);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void modifyParameter(QString value);

protected:
    QVariant m_value, m_min, m_max, m_step;
    QVariant m_valueold, m_minold, m_maxold, m_stepold;

    void removeFromControlPanel();
    void makeStepper(layoutType type, QWidget *w);
    void makeControlLine(layoutType type, QWidget *w);
    void makeModuleLine(layoutType type, QWidget *w);

    MELineEdit *m_textField;
    QVector<MELineEdit *> m_editList;
};
#endif
