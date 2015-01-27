/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_BOOLEANPORT_H
#define ME_BOOLEANPORT_H

#include "ports/MEParameterPort.h"

class QStringList;
class QString;
class QWidget;

class MECheckBox;
class MEControlParameterLine;
class MEModuleParameterLine;
class MEControlParameter;

//================================================
class MEBooleanPort : public MEParameterPort
//================================================
{
    Q_OBJECT

public:
    MEBooleanPort(MENode *node, QGraphicsScene *scene, const QString &pportname, const QString &paramtype, const QString &description);
    MEBooleanPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEBooleanPort();

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

    bool getBoolValue()
    {
        return m_value;
    }
    void setBoolValue(bool state)
    {
        m_value = state;
    }

private slots:

    void booleanCB();

private:
    bool m_value, m_valueold;
    MECheckBox *m_checkBox[2];

    void removeFromControlPanel();
};
#endif
