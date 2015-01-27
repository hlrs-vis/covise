/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_STRINGPORT_H
#define ME_STRINGPORT_H

#include "ports/MEParameterPort.h"

class QStringList;
class QString;
class QWidget;

class MEControlParameterLine;
class MEModuleParameterLine;
class MEControlParameter;
class MELineEdit;

//================================================
class MEStringPort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEStringPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &ptype, const QString &description);
    MEStringPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEStringPort();

#ifdef YAC
    void setValues(covise::coRecvBuffer &);
#endif

    void restoreParam();
    void storeParam();
    void defineParam(QString value, int apptype);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void modifyParameter(QString value);
    void sendParamMessage();
    void moduleParameterRequest();
    void makeLayout(layoutType, QWidget *);

private slots:

    void textCB(const QString &);

private:
    QString svalue, svalueold;

    void removeFromControlPanel();

    MELineEdit *editLine[2];
};
#endif
