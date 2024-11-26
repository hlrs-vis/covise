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
    if (appearanceType == STEPPER && type == CONTROL)
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
}

//!
//!  value changed
//!
void MEIntScalarPort::textCB(const QString &)
{

    QString text = m_editList.at(0)->text();
    m_value.setValue(text);
    m_step.setValue(m_editList.at(1)->text());
    sendParamMessage(text);

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//!  value changed
//!
void MEIntScalarPort::text2CB(const QString &text)
{
    m_value.setValue(text);
    sendParamMessage(text);

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEIntScalarPort::plusNewValue()
{
    m_value = m_value.toInt() + m_step.toInt();
    sendParamMessage(m_value.toString());

    // inform parent widget that value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEIntScalarPort::minusNewValue()
{
    m_value = m_value.toInt() - m_step.toInt();
    sendParamMessage(m_value.toString());

    // inform parent widget that value has been changed
    node->setModified(true);
}
