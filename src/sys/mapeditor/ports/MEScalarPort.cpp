/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>

#include "MEScalarPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"

/*!
    \class MEScalarPort
    \brief Base class for parameters of type SCALAR
*/

MEScalarPort::MEScalarPort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           const QString &paramtype,
                           const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
{
    m_textField = NULL;
}

MEScalarPort::MEScalarPort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           int paramtype,
                           const QString &description,
                           int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
{
    m_textField = NULL;
}

MEScalarPort::~MEScalarPort()
{
}

//!
//! restore saved parameters after after user has pressed cancel in module parameter window
//!
void MEScalarPort::restoreParam()
{
    m_value = m_valueold;
    sendParamMessage(toString(m_value));
}

//!
//! save current value for further use
//!
void MEScalarPort::storeParam()
{
    m_valueold = m_value;
}

//!
//! module has requested the parameter
//!
void MEScalarPort::moduleParameterRequest()
{
    sendParamMessage(toString(m_value));
}

//!
//! define parameter values (COVISE)
//!
void MEScalarPort::defineParam(QString svalue, int apptype)
{
    m_value.setValue(svalue);
    m_step.setValue(1);
    MEParameterPort::defineParam(svalue, apptype);
}

//!
//! modify parameter values (COVISE)
//!
void MEScalarPort::modifyParam(QStringList list, int noOfValues, int istart)
{
    Q_UNUSED(noOfValues);

    m_value.setValue(list[istart]);

    // modify module & control line content

    if (m_textField)
        m_textField->setText(toString(m_value));

    if (!m_editList.isEmpty())
    {
        m_editList.at(0)->setText(toString(m_value));
        m_editList.at(1)->setText(toString(m_step));
    }
}

//!
//! modify parameter values (COVISE)
//!
void MEScalarPort::modifyParameter(QString lvalue)
{
    m_value.setValue(lvalue);

    // modify module & control line content

    if (m_textField)
        m_textField->setText(lvalue);

    if (!m_editList.isEmpty())
    {
        m_editList.at(0)->setText(lvalue);
        m_editList.at(1)->setText(toString(m_step));
    }
}

//!
//! unmap parameter from control panel
//!
void MEScalarPort::removeFromControlPanel()
{
    MEParameterPort::removeFromControlPanel();
    m_textField = NULL;
}

#define addValue(text, m_value)                                                                \
    l = new QLabel(text, secondLine);                                                          \
    l->setFixedWidth(45);                                                                      \
    hb2->addWidget(l);                                                                         \
    le = new MELineEdit(secondLine);                                                           \
    le->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());                          \
    le->setText(toString(m_value));                                                           \
    connect(le, SIGNAL(contentChanged(const QString &)), this, SLOT(textCB(const QString &))); \
    hb2->addWidget(le, 4);                                                                     \
    m_editList << le;

//!
//! make a control line widget for the control panel
//!
void MEScalarPort::makeControlLine(layoutType, QWidget *w)
{
    // create a vertical layout for 2 rows

    QHBoxLayout *vb = new QHBoxLayout(w);
    vb->setMargin(2);
    vb->setSpacing(2);

    // text editor line

    m_textField = new MELineEdit(w);
    m_textField->setText(toString(m_value));
    m_textField->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());
    connect(m_textField, SIGNAL(contentChanged(const QString &)), this, SLOT(text2CB(const QString &)));
    connect(m_textField, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    vb->addWidget(m_textField, 6);
}

//!
//! make a module linewidget for parameter window,  second line contains details
//!
void MEScalarPort::makeModuleLine(layoutType, QWidget *w)
{
    // create a vertical layout for 2 rows

    QHBoxLayout *vb = new QHBoxLayout(w);
    vb->setMargin(2);
    vb->setSpacing(2);

    // create two container widgets

    QWidget *firstLine = new QWidget(w);
    secondLine = new QWidget(w);
    secondLine->hide();
    vb->addWidget(firstLine);
    vb->addWidget(secondLine);

    // create for each widget a horizontal layout

    QHBoxLayout *hb1 = new QHBoxLayout(firstLine);
    hb1->setMargin(2);
    hb1->setSpacing(2);

    QHBoxLayout *hb2 = new QHBoxLayout(secondLine);
    hb2->setMargin(2);
    hb2->setSpacing(2);

    // text editor line

    MELineEdit *le = new MELineEdit(firstLine);
    le->setText(toString(m_value));
    le->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());
    connect(le, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    connect(le, SIGNAL(contentChanged(const QString &)), this, SLOT(textCB(const QString &)));

    m_editList << le;
    hb1->addWidget(le, 6);

    QLabel *l;
    addValue("Step", m_step);
}

//!
//! make a stepper layout
//!
void MEScalarPort::makeStepper(layoutType, QWidget *w)
{

    // horizontal Layout

    QHBoxLayout *hbox = new QHBoxLayout(w);
    hbox->setMargin(2);
    hbox->setSpacing(2);

    // text filed with current value

    m_textField = new MELineEdit(w);
    m_textField->setText(toString(m_value));
    m_textField->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());

    connect(m_textField, SIGNAL(contentChanged(const QString &)), this, SLOT(text2CB(const QString &)));
    connect(m_textField, SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    hbox->addWidget(m_textField, 2);

    // left arrow button

    left1 = new QPushButton(w);
    left1->setIcon(QPixmap(":/icons/1leftarrow.png"));
    left1->setFlat(true);
    connect(left1, SIGNAL(clicked()), this, SLOT(left1CB()));
    connect(left1, SIGNAL(clicked()), left1, SLOT(setFocus()));
    hbox->addWidget(left1);

    // left arrow button

    right1 = new QPushButton(w);
    right1->setIcon(QPixmap(":/icons/1rightarrow.png"));
    right1->setFlat(true);
    connect(right1, SIGNAL(clicked()), this, SLOT(right1CB()));
    connect(right1, SIGNAL(clicked()), left1, SLOT(setFocus()));
    hbox->addWidget(right1);
}
