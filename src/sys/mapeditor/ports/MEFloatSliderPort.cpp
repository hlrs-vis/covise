/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <climits>
#include <QSlider>

#include "MEFloatSliderPort.h"
#include "MELineEdit.h"
#include "nodes/MENode.h"
#include "widgets/MEUserInterface.h"

/*!
    \class MEFloatSliderPort
    \brief Class handles float slider parameters
*/

MEFloatSliderPort::MEFloatSliderPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     const QString &paramtype,
                                     const QString &description)
    : MESliderPort(node, scene, portname, paramtype, description)
{
}

MEFloatSliderPort::MEFloatSliderPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     int paramtype,
                                     const QString &description,
                                     int porttype)
    : MESliderPort(node, scene, portname, paramtype, description, porttype)
{
}

MEFloatSliderPort::~MEFloatSliderPort()
{
}

//!
//! decide wich layout should be created
//!
void MEFloatSliderPort::makeLayout(layoutType type, QWidget *w)
{

    if (appearanceType == STEPPER && type == CONTROL)
        makePlayer(type, w, toString(m_value));

    else if (type == CONTROL)
        makeControlLine(type, w, toString(m_value));

    else if (type == MODULE)
    {
        QStringList buffer;
        buffer << toString(m_min) << toString(m_max) << toString(m_value) << toString(m_step);
        makeModuleLine(type, w, buffer);
    }
}

//!
//! create slider widget
//!
QSlider *MEFloatSliderPort::makeSlider(QWidget *parent)
{
    QSlider *m_slider = new QSlider(Qt::Horizontal, parent);
    int rangemin = 0;
    int rangemax = INT_MAX;
    int sliderstep = int(m_step / (m_max - m_min) * INT_MAX);
    int slidervalue = fToI(m_value);

    m_slider->setRange(rangemin, rangemax);
    m_slider->setValue(slidervalue);
    m_slider->setSingleStep(sliderstep);
    m_slider->setTracking(true);
    m_slider->setFocusPolicy(Qt::WheelFocus);

    connect(m_slider, SIGNAL(sliderReleased()), this, SLOT(slider2CB()));
    connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(slider1CB(int)));

    return m_slider;
}

//!
//! restore saved parameters after after user has pressed cancel in module parameter window
//!
void MEFloatSliderPort::restoreParam()
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
void MEFloatSliderPort::storeParam()
{
    m_minold = m_min;
    m_maxold = m_max;
    m_valueold = m_value;
    m_stepold = m_step;
}

//!
//! module has requested the parameter
//!
void MEFloatSliderPort::moduleParameterRequest()
{
    sendParamMessage();
}

//!
//! define parameter values (COVISE)
//!
void MEFloatSliderPort::defineParam(QString svalue, int apptype)
{
    QStringList list = svalue.split(" ", QString::SkipEmptyParts);

    m_min = list[0].toFloat();
    m_max = list[1].toFloat();
    m_value = list[2].toFloat();
    m_step = qAbs(m_max - m_min) / 30.;

    MEParameterPort::defineParam(svalue, apptype);
}

//!
//! modify parameter values (COVISE)
//!
void MEFloatSliderPort::modifyParam(QStringList list, int noOfValues, int istart)
{
    Q_UNUSED(noOfValues);

    if (list.count() > istart + 2)
    {
        m_min = list[istart].toFloat();
        m_max = list[istart + 1].toFloat();
        m_value = list[istart + 2].toFloat();

        if (!m_editList.isEmpty())
        {
            m_editList.at(0)->setText(toString(m_value));
            m_editList.at(1)->setText(toString(m_min));
            m_editList.at(2)->setText(toString(m_max));
        }

        // modify module & control line content
        if (m_textField)
            m_textField->setText(toString(m_value));

        if (m_slider[MODULE])
        {
            int ival = fToI(m_value);
            m_slider[MODULE]->setValue(ival);
        }

        if (m_slider[CONTROL])
        {
            int ival = fToI(m_value);
            m_slider[CONTROL]->setValue(ival);
        }
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
}

//!
//! modify parameter values (COVISE)
//!
void MEFloatSliderPort::modifyParameter(QString lvalue)
{
    QStringList list = QString(lvalue).split(" ");

    if (list.count() == 3)
    {
        m_min = list[0].toFloat();
        m_max = list[1].toFloat();
        m_value = list[2].toFloat();

        if (!m_editList.isEmpty())
        {
            m_editList.at(0)->setText(toString(m_value));
            m_editList.at(1)->setText(toString(m_min));
            m_editList.at(2)->setText(toString(m_max));
        }

        // modify module & control line content
        if (m_textField)
            m_textField->setText(toString(m_value));

        if (m_slider[MODULE])
        {
            int ival = fToI(m_value);
            m_slider[MODULE]->setValue(ival);
        }

        if (m_slider[CONTROL])
        {
            int ival = fToI(m_value);
            m_slider[CONTROL]->setValue(ival);
        }
    }

    else
    {
        QString msg = "MEParameterPort::modifyParam: " + node->getNodeTitle() + ": Parameter type " + parameterType + " has wrong number of values";
        MEUserInterface::instance()->printMessage(msg);
    }
}

//!
//! adapt  boundaries
//!
void MEFloatSliderPort::boundaryCB()
{
    float m_value = m_editList.at(0)->text().toFloat();
    float m_min = m_editList.at(1)->text().toFloat();
    float m_max = m_editList.at(2)->text().toFloat();
    float m_step = m_editList.at(3)->text().toFloat();
    m_value = qMax(m_value, m_min);
    m_value = qMin(m_value, m_max);

    int istep = fToI(m_step);

    if (m_slider[MODULE])
        m_slider[MODULE]->setSingleStep(istep);

    if (m_slider[CONTROL])
        m_slider[CONTROL]->setSingleStep(istep);

    sendParamMessage();
}

//!
//!  value changed
//!
void MEFloatSliderPort::textCB(const QString &)
{

    m_value = m_editList.at(0)->text().toFloat();
    m_min = m_editList.at(1)->text().toFloat();
    m_max = m_editList.at(2)->text().toFloat();
    m_step = m_editList.at(3)->text().toFloat();

    sendParamMessage();

    // inform parent widget that m_value has been changed
    node->setModified(true);
}

//!
//!  value changed
//!
void MEFloatSliderPort::text2CB(const QString &text)
{

    m_value = text.toFloat();
    sendParamMessage();

    // inform parent widget that m_value has been changed
    node->setModified(true);
}

//!
//! PlayerCB
//!
void MEFloatSliderPort::plusNewValue()
{

    float ff = m_value + m_step;
    m_value = qMin(ff, m_max);
    m_value = qMax(ff, m_min);

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
void MEFloatSliderPort::minusNewValue()
{

    float ff = m_value - m_step;
    m_value = qMin(ff, m_max);
    m_value = qMax(ff, m_min);

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
void MEFloatSliderPort::sendParamMessage()
{

    QString buffer;
    buffer = toString(m_min) + " " + toString(m_max) + " " + toString(m_value);

    MEParameterPort::sendParamMessage(buffer);
}

//!
//! transform a given float value to integer
//!
int MEFloatSliderPort::fToI(float fvalue)
{
    float ff = (fvalue - m_min) / (m_max - m_min);
    int ival = int(INT_MAX * ff);
    return ival;
}

//!
//! transform a given integer value to float
//!
float MEFloatSliderPort::iToF(int ivalue)
{
    float fval = m_min + (float(ivalue) / INT_MAX * (m_max - m_min));
    return fval;
}

//!
//! get the new slider value ( slider was released after dragging)
//!
void MEFloatSliderPort::slider2CB()
{
    // object that sent the signal

    const QObject *obj = sender();
    QSlider *sl = (QSlider *)obj;

    // get the value

    int val;
    if (sl == m_slider[MODULE])
        val = m_slider[MODULE]->value();
    else
        val = m_slider[CONTROL]->value();

    float ff = iToF(val);
    if (ff == m_value)
        return;

    m_value = ff;
    sendParamMessage();

    // inform parent widget that m_value has been changed

    node->setModified(true);
}

//!
//! get the new slider value ( update only the text line for current value)
//!
void MEFloatSliderPort::slider1CB(int val)
{
    // object that sent the signal

    //const QObject *obj = sender();
    //QSlider *sl = (QSlider*) obj;

    // update text line

    //int type = MODULE;
    //if(sl == m_slider[CONTROL])
    //   type = CONTROL;

    float ff = iToF(val);
    if (ff == m_value)
        return;

    if (m_textField)
        m_textField->setText(toString(ff));

    if (!m_editList.isEmpty())
        m_editList.at(0)->setText(toString(ff));

    // inform parent widget that m_value has been changed

    node->setModified(true);
}
