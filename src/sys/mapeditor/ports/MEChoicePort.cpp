/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QHBoxLayout>
#include <QDebug>

#include "MEChoicePort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "widgets/MEUserInterface.h"
#include "nodes/MENode.h"

/*!
    \class MEChoicePort
    \brief Class for parameter port of type ::CHOICE
*/

MEChoicePort::MEChoicePort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           const QString &paramtype,
                           const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
{
    m_comboBox[MODULE] = m_comboBox[CONTROL] = NULL;
}

MEChoicePort::MEChoicePort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           int paramtype,
                           const QString &description,
                           int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
{
    m_comboBox[MODULE] = m_comboBox[CONTROL] = NULL;
}

MEChoicePort::~MEChoicePort()
{
}

//!
//! Restore saved parameters after, user has pressed cancel in module parameter window
//!
void MEChoicePort::restoreParam()
{
    m_currentChoice = m_currentChoiceold;
    m_choiceValues = m_choiceValuesold;
    sendParamMessage();
}

//!
//! Save current value for further use
//!
void MEChoicePort::storeParam()
{
    m_currentChoiceold = m_currentChoice;
    m_choiceValuesold = m_choiceValues;
}

//!
//! module has requested parameter
//!
void MEChoicePort::moduleParameterRequest()
{
    sendParamMessage();
}

//!
//! Define one parameter
//!
void MEChoicePort::defineParam(QString value, int apptype)
{
#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    // choices - send only current index of list
    // choices - send the complete list
    QStringList list = value.split(" ", QString::SkipEmptyParts);
    int count = list.count();

    // current index to choice list
    m_currentChoice = list[0].toInt() - 1;

    // replace '177' to allow blanks in choice label
    // store choices inside a list
    for (int j = 1; j < count; j++)
    {
        QString tmp = list[j];
        QString text = tmp.replace(QChar('\177'), " ");
        m_choiceValues.append(text);
    }

    // check
    if (m_currentChoice >= m_choiceValues.count())
    {
        QString text = "_________ ATTENTION: " + node->getTitle() + "::" + portname;
        text.append(";  No. of choices is " + QString::number(m_choiceValues.count()) + ", current no. is " + QString::number(m_currentChoice));
        MEUserInterface::instance()->printMessage(text);
    }

    MEParameterPort::defineParam(value, apptype);
#endif
}

//!
//! Modify one parameter
//!
void MEChoicePort::modifyParam(QStringList list, int noOfValues, int istart)
{
#ifdef YAC

    Q_UNUSED(list);
    Q_UNUSED(istart);
    Q_UNUSED(noOfValues);

#else
    Q_UNUSED(noOfValues);

    // this is an old fashioned parameter message
    // use only new index into list
    m_currentChoice = list[istart].toInt() - 1;

    if (noOfValues == 2) // number and values
    {
        QString values = list[istart + 1].trimmed();
        QStringList list2 = values.split(" ");

        if (list2.count() > 0)
        {

            // replace '177' to allow blanks in choice label
            // store choices inside a list
            m_choiceValues.clear();

            for (int j = 0; j < list2.count(); j++)
            {
                QString text = list2[j].replace(QChar('\177'), " ");
                m_choiceValues.append(text);
            }

            if (m_comboBox[MODULE])
            {
                m_comboBox[MODULE]->clear();
                m_comboBox[MODULE]->addItems(m_choiceValues);
                //m_comboBox[MODULE]->setMinimumContentsLength(20);
            }

            if (m_comboBox[CONTROL])
            {
                m_comboBox[CONTROL]->clear();
                m_comboBox[CONTROL]->addItems(m_choiceValues);
                //m_comboBox[CONTROL]->setMinimumContentsLength(20);
            }
        }
    }

    // check current index
    if (m_currentChoice >= m_choiceValues.count())
    {
        QString text = "_________ ATTENTION: " + node->getTitle() + "::" + portname;
        text.append(";  No. of choices is " + QString::number(m_choiceValues.count()) + ", current no. is " + QString::number(m_currentChoice));
        MEUserInterface::instance()->printMessage(text);
    }

    // modify module & control line content
    if (m_comboBox[MODULE])
        m_comboBox[MODULE]->setCurrentIndex(m_currentChoice);

    if (m_comboBox[CONTROL])
        m_comboBox[CONTROL]->setCurrentIndex(m_currentChoice);
#endif
}

//!
//! Modify one parameter
//!
void MEChoicePort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    QString values = lvalue.trimmed();
    QStringList list = values.split(" ");

    m_currentChoice = list[0].toInt() - 1;

    if (list.count() > 1)
    {

        // replace '177' to allow blanks in choice label
        // store choices inside a list
        m_choiceValues.clear();

        for (int j = 1; j < list.count(); j++)
        {
            QString text = list[j].replace(QChar('\177'), " ");
            m_choiceValues.append(text);
        }

        if (m_comboBox[MODULE])
        {
            m_comboBox[MODULE]->clear();
            m_comboBox[MODULE]->addItems(m_choiceValues);
        }

        if (m_comboBox[CONTROL])
        {
            m_comboBox[CONTROL]->clear();
            m_comboBox[CONTROL]->addItems(m_choiceValues);
        }
    }

    if (m_currentChoice > list.count() - 2)
    {
        QString text = "_________ ATTENTION: " + node->getTitle() + "::" + portname;
        text.append(";  No. of choices is " + QString::number(m_choiceValues.count()) + ", current no. is " + QString::number(m_currentChoice));
        MEUserInterface::instance()->printMessage(text);
    }

    else
    {
        // modify module & control line content
        if (m_comboBox[MODULE])
            m_comboBox[MODULE]->setCurrentIndex(m_currentChoice);

        if (m_comboBox[CONTROL])
            m_comboBox[CONTROL]->setCurrentIndex(m_currentChoice);
    }
#endif
}

//!
//! Unmap parameter from control panel
//!
void MEChoicePort::removeFromControlPanel()
{
    MEParameterPort::removeFromControlPanel();
    m_comboBox[CONTROL] = NULL;
}

//!
//! Create the layout for the module parameter widget
//!
void MEChoicePort::makeLayout(layoutType type, QWidget *w)
{
    if (m_comboBox[type] != NULL)
    {
        qCritical() << "_________ ATTENTION: makeLayout widget already created" << portname;
        return;
    }

    QHBoxLayout *hBox = new QHBoxLayout(w);
    hBox->setMargin(2);
    hBox->setSpacing(2);

    m_comboBox[type] = new MEComboBox();
    m_comboBox[type]->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    m_comboBox[type]->addItems(m_choiceValues);
    m_comboBox[type]->setCurrentIndex(m_currentChoice);
    connect(m_comboBox[type], SIGNAL(activated(int)), this, SLOT(choiceCB(int)));
    connect(m_comboBox[type], SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));

    hBox->addWidget(m_comboBox[type]);
    hBox->addStretch(0);
}

//!
//! Send a PARAM message to controller
//!
void MEChoicePort::sendParamMessage()
{

    QString mybuffer;

    mybuffer.append(QString::number(m_currentChoice + 1));

    QChar sep = '\177';
    for (unsigned int j = 0; j < m_choiceValues.count(); j++)
    {
        QString text = m_choiceValues[j].replace(" ", sep);
        mybuffer.append(" " + text);
    }

    MEParameterPort::sendParamMessage(mybuffer);
}

//!
//! Get the new choice value
//!
void MEChoicePort::choiceCB(int state)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << 0 << state;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_currentChoice = state;
    sendParamMessage();
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC

//!
//! Set new values
//!
void MEChoicePort::setValues(covise::coRecvBuffer &tb)
{
    int flag;
    const char *name;
    tb >> flag;

    // new values
    if (flag)
    {
        tb >> m_noOfChoices;
        if (!m_choiceValues.isEmpty())
            m_choiceValues.clear();

        for (int j = 0; j < m_noOfChoices; j++)
        {
            tb >> name;
            m_choiceValues << name;
        }
    }

    // new selection
    tb >> m_currentChoice;
}
#endif
