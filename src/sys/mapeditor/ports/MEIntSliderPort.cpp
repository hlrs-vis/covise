/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QSlider>

#include "MEIntSliderPort.h"
#include "MELineEdit.h"
#include "nodes/MENode.h"
#include "widgets/MEUserInterface.h"

/*!
    \class MEIntSliderPort
    \brief Class handles integer slider parameters
*/

MEIntSliderPort::MEIntSliderPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 const QString &paramtype,
                                 const QString &description)
    : MESliderPort(node, scene, portname, paramtype, description)
{
}

MEIntSliderPort::MEIntSliderPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 int paramtype,
                                 const QString &description,
                                 int porttype)
    : MESliderPort(node, scene, portname, paramtype, description, porttype)
{
}

MEIntSliderPort::~MEIntSliderPort()
{
}

//!
//! decide wich layout should be created
//!
void MEIntSliderPort::makeLayout(layoutType type, QWidget *w)
{

#ifdef YAC
    if (appearanceType == A_STEPPER && type == CONTROL)
#else
    if (appearanceType == STEPPER && type == CONTROL)
#endif
        makePlayer(type, w, QString::number(m_value));

    else if (type == CONTROL)
        makeControlLine(type, w, QString::number(m_value));

    else if (type == MODULE)
    {
        QStringList buffer;
        buffer << QString::number(m_min) << QString::number(m_max) << QString::number(m_value) << QString::number(m_step);
        makeModuleLine(type, w, buffer);
    }
}

//!
//! create slider widget
//!
QSlider *MEIntSliderPort::makeSlider(QWidget *w)
{

    QSlider *m_slider = new QSlider(Qt::Horizontal, w);
    m_slider->setRange(m_min, m_max);
    m_slider->setValue(m_value);
    m_slider->setSingleStep(m_step);
    m_slider->setTracking(true);
    m_slider->setFocusPolicy(Qt::WheelFocus);

    connect(m_slider, SIGNAL(sliderReleased()), this, SLOT(slider2CB()));
    connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(slider1CB(int)));

    return m_slider;
}

//!
//! restore saved parameters after after user has pressed cancel in module parameter window
//!
void MEIntSliderPort::restoreParam()
{
    m_min = m_minold;
    m_max = m_maxold;
    m_value = m_valueold;
    m_step = m_stepold;
    sendParamMessage();
}

//!
//! save current value for further use
//!
void MEIntSliderPort::storeParam()
{
    m_minold = m_min;
    m_maxold = m_max;
    m_valueold = m_value;
    m_stepold = m_step;
}

//!
//! module has requested the parameter
//!
void MEIntSliderPort::moduleParameterRequest()
{
    sendParamMessage();
}

//!
//! define parameter values (COVISE)
//!
void MEIntSliderPort::defineParam(QString svalue, int apptype)
{
#ifdef YAC

    Q_UNUSED(svalue);
    Q_UNUSED(apptype);

#else

    QStringList list = svalue.split(" ", QString::SkipEmptyParts);

    m_min = list[0].toLong();
    m_max = list[1].toLong();
    m_value = list[2].toLong();
    m_step = 1;

    MEParameterPort::defineParam(svalue, apptype);
#endif
}

//!
//! modify parameter values (COVISE)
//!
void MEIntSliderPort::modifyParam(QStringList list, int noOfValues, int istart)
{
#ifdef YAC

    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);

#else

    Q_UNUSED(noOfValues);

    if (list.count() > istart + 2)
    {
        m_min = list[istart].toLong();
        m_max = list[istart + 1].toLong();
        m_value = list[istart + 2].toLong();

        if (!m_editList.isEmpty())
        {
            m_editList.at(0)->setText(QString::number(m_value));
            m_editList.at(1)->setText(QString::number(m_min));
            m_editList.at(2)->setText(QString::number(m_max));
        }

        // modify module & control line content
        if (m_textField)
            m_textField->setText(QString::number(m_value));

        if (m_slider[MODULE])
            m_slider[MODULE]->setValue(m_value);

        if (m_slider[CONTROL])
            m_slider[CONTROL]->setValue(m_value);
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
#endif
}

//!
//! modify parameter values (COVISE)
//!
void MEIntSliderPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    QStringList list = QString(lvalue).split(" ");

    if (list.count() == 3)
    {
        m_min = list[0].toLong();
        m_max = list[1].toLong();
        m_value = list[2].toLong();

        if (!m_editList.isEmpty())
        {
            m_editList.at(0)->setText(QString::number(m_value));
            m_editList.at(1)->setText(QString::number(m_min));
            m_editList.at(2)->setText(QString::number(m_max));
        }

        // modify module & control line content
        if (m_textField)
            m_textField->setText(QString::number(m_value));

        if (m_slider[MODULE])
            m_slider[MODULE]->setValue(m_value);

        if (m_slider[CONTROL])
            m_slider[CONTROL]->setValue(m_value);
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
#endif
}

//!
//! adapt  boundaries
//!
void MEIntSliderPort::boundaryCB()
{
    long m_value = m_editList.at(0)->text().toLong();
    long m_min = m_editList.at(1)->text().toLong();
    long m_max = m_editList.at(2)->text().toLong();
    long m_step = m_editList.at(3)->text().toLong();
    m_value = qMax(m_value, m_min);
    m_value = qMin(m_value, m_max);

    if (m_slider[MODULE])
        m_slider[MODULE]->setSingleStep(m_step);

    if (m_slider[CONTROL])
        m_slider[CONTROL]->setSingleStep(m_step);

    sendParamMessage();
}

//!
//!  value changed
//!
void MEIntSliderPort::textCB(const QString &)
{

    m_value = m_editList.at(0)->text().toLong();
    m_min = m_editList.at(1)->text().toLong();
    m_max = m_editList.at(2)->text().toLong();
    m_step = m_editList.at(3)->text().toLong();

    sendParamMessage();

    // inform parent widget that m_value has been changed
    node->setModified(true);
}

//!
//!  value changed
//!
void MEIntSliderPort::text2CB(const QString &text)
{
    m_value = text.toLong();
    sendParamMessage();

    // inform parent widget that m_value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEIntSliderPort::plusNewValue()
{

    m_value = m_value + m_step;
    m_value = qMin(m_value, m_max);
    m_value = qMax(m_value, m_min);

    if (timer)
    {
        if (m_value == m_max || m_value == m_min)
            stopCB();
    }

    sendParamMessage();

    // inform parent widget that m_value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEIntSliderPort::minusNewValue()
{

    m_value = m_value - m_step;
    m_value = qMin(m_value, m_max);
    m_value = qMax(m_value, m_min);

    if (timer)
    {
        if (m_value == m_max || m_value == m_min)
            stopCB();
    }

    sendParamMessage();

    // inform parent widget that m_value has been changed
    node->setModified(true);
}

//!
//! send a PARAM message to controller
//!
void MEIntSliderPort::sendParamMessage()
{
    QString buffer;
    buffer = QString::number(m_min) + " " + QString::number(m_max) + " " + QString::number(m_value);
    MEParameterPort::sendParamMessage(buffer);
}

//!
//! get the new slider value ( slider was released after dragging)
//!
void MEIntSliderPort::slider2CB()
{

    // find widget that send the data

    const QObject *obj = sender();
    QSlider *sl = (QSlider *)obj;

    // get new value

    int val;
    if (sl == m_slider[MODULE])
        val = m_slider[MODULE]->value();
    else
        val = m_slider[CONTROL]->value();

    if (val == m_value)
        return;

    m_value = val;
    sendParamMessage();

    // inform parent widget that m_value has been changed

    node->setModified(true);
}

//!
//! get the new slider value ( update only the text line for current value)
//!
void MEIntSliderPort::slider1CB(int val)
{

    // find widget that send the  in list

    //const QObject *obj = sender();
    //QSlider *sl = (QSlider*) obj;

    // update text line

    //int type = MODULE;
    //if(sl == m_slider[CONTROL])
    //   type = CONTROL;

    if (val == m_value)
        return;

    if (m_textField)
        m_textField->setText(QString::number(val));

    if (!m_editList.isEmpty())
        m_editList.at(0)->setText(QString::number(val));

    // inform parent widget that m_value has been changed

    node->setModified(true);
}

#ifdef YAC
void MEIntSliderPort::setValues(covise::coRecvBuffer &)
{
}
#endif
