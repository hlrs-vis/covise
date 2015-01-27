/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QHBoxLayout>

#include "MEBooleanPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"

/*!
    \class MEBooleanPort
    \brief Class for parameter port of type ::BOOLEAN
*/

MEBooleanPort::MEBooleanPort(MENode *node, QGraphicsScene *scene,
                             const QString &portname,
                             const QString &paramtype,
                             const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
{
    m_checkBox[MODULE] = m_checkBox[CONTROL] = NULL;
}

MEBooleanPort::MEBooleanPort(MENode *node, QGraphicsScene *scene,
                             const QString &portname,
                             int paramtype,
                             const QString &description,
                             int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
{
    m_checkBox[MODULE] = m_checkBox[CONTROL] = NULL;
}

MEBooleanPort::~MEBooleanPort()
{
}

//!
//! Restore saved parameters after, the user has pressed cancel in module parameter window
//!
void MEBooleanPort::restoreParam()
{
    m_value = m_valueold;
    sendParamMessage();
}

//!
//! Save current value for further use
//!
void MEBooleanPort::storeParam()
{
    m_valueold = m_value;
}

//!
//! module has requested parameter
//!
void MEBooleanPort::moduleParameterRequest()
{
    sendParamMessage();
}

//!
//! Define one parameter, called from controller
//!
void MEBooleanPort::defineParam(QString value, int apptype)
{
#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    QString myVal = value.toUpper();
    if (myVal == "TRUE")
        m_value = true;
    else
        m_value = false;

    MEParameterPort::defineParam(value, apptype);
#endif
}

//!
//! Modify one parameter, update from controller
//!
void MEBooleanPort::modifyParam(QStringList list, int noOfValues, int istart)
{
#ifdef YAC

    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);

#else

    Q_UNUSED(noOfValues);

    QString myVal = list[istart].toUpper();
    if (myVal == "TRUE")
        m_value = true;
    else
        m_value = false;

    // modify module & control line content
    if (m_checkBox[MODULE])
        m_checkBox[MODULE]->setChecked(m_value);

    if (m_checkBox[CONTROL])
        m_checkBox[CONTROL]->setChecked(m_value);
#endif
}

//!
//! Modify one parameter, update from controller
//!
void MEBooleanPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    QString myVal = lvalue.toUpper();
    if (myVal == "TRUE")
        m_value = true;
    else
        m_value = false;

    // modify module & control line content
    if (m_checkBox[MODULE])
        m_checkBox[MODULE]->setChecked(m_value);

    if (m_checkBox[CONTROL])
        m_checkBox[CONTROL]->setChecked(m_value);
#endif
}

//!
//! Unmap parameter from control panel
//!
void MEBooleanPort::removeFromControlPanel()
{
    MEParameterPort::removeFromControlPanel();
    m_checkBox[CONTROL] = NULL;
}

//!
//! Create layout for the module parameter widget or control panel
//!
void MEBooleanPort::makeLayout(layoutType type, QWidget *w)
{
    QHBoxLayout *hBox = new QHBoxLayout(w);
    hBox->setMargin(2);
    hBox->setSpacing(2);

    m_checkBox[type] = new MECheckBox();
    m_checkBox[type]->setChecked(m_value);
    connect(m_checkBox[type], SIGNAL(clicked()), this, SLOT(booleanCB()));
    connect(m_checkBox[type], SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));

    hBox->addWidget(m_checkBox[type]);
    hBox->addStretch(4);
}

//!
//! Send a PARAM message to controller
//!
void MEBooleanPort::sendParamMessage()
{
    QString mybuffer;

    if (m_value)
        mybuffer = "TRUE";
    else
        mybuffer = "FALSE";

    MEParameterPort::sendParamMessage(mybuffer);
}

//!
//! Get the new value & send it to the controller
//!
void MEBooleanPort::booleanCB()
{

    // object that sent the signal
    const QObject *obj = sender();
    MECheckBox *cb = (MECheckBox *)obj;

    // find widget that send the  in list
    if (cb == m_checkBox[MODULE])
        m_value = m_checkBox[MODULE]->isChecked();
    else
        m_value = m_checkBox[CONTROL]->isChecked();

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << m_value;

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    sendParamMessage();
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC

//!
//! Set new values, only used by YAC
void MEBooleanPort::setValues(covise::coRecvBuffer &tb)
{
    bool flag;

    tb >> flag;
    m_value = flag;

    // modify module & control line content
    if (m_checkBox[MODULE])
        m_checkBox[MODULE]->setChecked(m_value);

    if (m_checkBox[CONTROL])
        m_checkBox[CONTROL]->setChecked(m_value);
}
#endif
