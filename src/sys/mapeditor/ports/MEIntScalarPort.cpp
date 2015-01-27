/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include "MEIntScalarPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"

;

/*!
    \class MEIntScalarPort
    \brief Class handles integer scalar parameters
*/

MEIntScalarPort::MEIntScalarPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 const QString &paramtype,
                                 const QString &description)
    : MEScalarPort(node, scene, portname, paramtype, description)
{
}

MEIntScalarPort::MEIntScalarPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 int paramtype,
                                 const QString &description,
                                 int porttype)
    : MEScalarPort(node, scene, portname, paramtype, description, porttype)
{
}

MEIntScalarPort::~MEIntScalarPort()
{
}

//!
//! decide wich layout should be created
//!
void MEIntScalarPort::makeLayout(layoutType type, QWidget *w)
{

// Stepper
#ifdef YAC
    if (appearanceType == A_STEPPER && type == CONTROL)
#else
    if (appearanceType == STEPPER && type == CONTROL)
#endif
        makeStepper(type, w);

    else if (type == CONTROL)
        makeControlLine(type, w);

    else if (type == MODULE)
        makeModuleLine(type, w);
}

//!
//! adapt boundaries  (YAC)
//!
void MEIntScalarPort::boundaryCB()
{
#ifdef YAC

    long val = m_editList.at(0)->text().toInt();
    long min = m_editList.at(1)->text().toInt();
    long max = m_editList.at(2)->text().toInt();
    long step = m_editList.at(3)->text().toInt();
    val = qMax(val, min);
    val = qMin(val, max);

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << val << min << max << step;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);
#endif
}

//!
//!  value changed
//!
void MEIntScalarPort::textCB(const QString &)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;

    for (int i = 0; i < 4; i++)
        sb << m_editList.at(i)->text().toInt();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    QString text = m_editList.at(0)->text();
    m_value.setValue(text);
    m_step.setValue(m_editList.at(1)->text());
    sendParamMessage(text);
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//!  value changed
//!
void MEIntScalarPort::text2CB(const QString &text)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << text.toInt();
    sb << m_min.toInt() << m_max.toInt() << m_step.toInt();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_value.setValue(text);
    sendParamMessage(text);
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEIntScalarPort::plusNewValue()
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << getNode()->getNodeID() << portname;

    int ii = m_value.toInt() + m_step.toInt();
    ii = qMin(ii, m_max.toInt());
    ii = qMax(ii, m_min.toInt());

    sb << ii << m_min.toInt() << m_max.toInt() << m_step.toInt();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_value = m_value.toInt() + m_step.toInt();
    sendParamMessage(m_value.toString());
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEIntScalarPort::minusNewValue()
{

#ifdef YAC

    int ii = m_value.toInt() - m_step.toInt();
    ii = qMin(ii, m_max.toInt());
    ii = qMax(ii, m_min.toInt());

    covise::coSendBuffer sb;
    sb << getNode()->getNodeID() << portname;
    sb << ii << m_min.toInt() << m_max.toInt() << m_step.toInt();
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_value = m_value.toInt() - m_step.toInt();
    sendParamMessage(m_value.toString());
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC

//!
//!  get new values
//!
void MEIntScalarPort::setValues(covise::coRecvBuffer &tb)
{
    int ff, fmin, fmax, fstep;
    tb >> ff >> fmin >> fmax >> fstep;

    m_value.setValue(ff);
    m_min.setValue(fmin);
    m_max.setValue(fmax);
    m_step.setValue(fstep);

    // modify module & control line content
    if (m_textField)
        m_textField->setText(QString::number(ff));

    if (!m_editList.isEmpty())
    {
        m_editList.at(0)->setText(QString::number(ff));
        m_editList.at(1)->setText(QString::number(fmin));
        m_editList.at(2)->setText(QString::number(fmax));
        m_editList.at(3)->setText(QString::number(fstep));
    }
}
#endif
