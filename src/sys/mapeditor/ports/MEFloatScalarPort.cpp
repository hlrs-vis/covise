/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include "MEFloatScalarPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"

;

/*!
    \class MEFloatScalarPort
    \brief Class handles float scalar parameters
*/

MEFloatScalarPort::MEFloatScalarPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     const QString &paramtype,
                                     const QString &description)
    : MEScalarPort(node, scene, portname, paramtype, description)
{
}

MEFloatScalarPort::MEFloatScalarPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     int paramtype,
                                     const QString &description,
                                     int porttype)
    : MEScalarPort(node, scene, portname, paramtype, description, porttype)
{
}

MEFloatScalarPort::~MEFloatScalarPort()
{
}

//!
//! decide wich layout should be created
//!
void MEFloatScalarPort::makeLayout(layoutType type, QWidget *w)
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
//! adapt  boundaries  (YAC)
//!
void MEFloatScalarPort::boundaryCB()
{
#ifdef YAC

    float val = m_editList.at(0)->text().toFloat();
    float min = m_editList.at(1)->text().toFloat();
    float max = m_editList.at(2)->text().toFloat();
    float step = m_editList.at(3)->text().toFloat();
    val = qMax(val, min);
    val = qMin(val, max);

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname << val << min << max << step;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);
#endif
}

//!
//! float value changed
//!
void MEFloatScalarPort::textCB(const QString &)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;

    for (int i = 0; i < 4; i++)
        sb << m_editList.at(i)->text().toFloat();

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
//!  float value changed
//!
void MEFloatScalarPort::text2CB(const QString &text)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << text.toDouble();
    sb << m_min.toDouble() << m_max.toDouble() << m_step.toDouble();
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
void MEFloatScalarPort::plusNewValue()
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << getNode()->getNodeID() << portname;

    double ff = m_value.toDouble() + m_step.toDouble();
    ff = qMin(ff, m_max.toDouble());
    ff = qMin(ff, m_min.toDouble());

    sb << ff << m_min.toDouble() << m_max.toDouble() << m_step.toDouble();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_value = m_value.toDouble() + m_step.toDouble();
    sendParamMessage(m_value.toString());
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEFloatScalarPort::minusNewValue()
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << getNode()->getNodeID() << portname;

    double ff = m_value.toDouble() - m_step.toDouble();
    ff = qMin(ff, m_max.toDouble());
    ff = qMin(ff, m_min.toDouble());

    sb << ff << m_min.toDouble() << m_max.toDouble() << m_step.toDouble();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    m_value = m_value.toDouble() - m_step.toDouble();
    sendParamMessage(m_value.toString());
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC

//!
//!  get new values
//!
void MEFloatScalarPort::setValues(covise::coRecvBuffer &tb)
{
    float ff, fmin, fmax, fstep;
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
