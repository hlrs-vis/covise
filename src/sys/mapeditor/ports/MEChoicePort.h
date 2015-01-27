/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_CHOICEPORT_H
#define ME_CHOICEPORT_H

#include "ports/MEParameterPort.h"

class QStringList;
class QString;
class QWidget;

class MEComboBox;
class MEControlParameterLine;
class MEModuleParameterLine;
class MEControlParameter;

//================================================
class MEChoicePort : public MEParameterPort
//================================================
{

    Q_OBJECT

public:
    MEChoicePort(MENode *node, QGraphicsScene *scene, const QString &portname, const QString &paramtype, const QString &description);
    MEChoicePort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEChoicePort();

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

    void choiceCB(int);

private:
    int m_noOfChoices;
    unsigned int m_currentChoice, m_currentChoiceold;
    QStringList m_choiceValues, m_choiceValuesold;

    MEComboBox *m_comboBox[2];

    void removeFromControlPanel();
};
#endif
