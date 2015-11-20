/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QPushButton>

#include <covise/covise_msg.h>

#include "MEParameterPort.h"
#include "METimer.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"
#include "modulePanel/MEModuleParameterLine.h"
#include "controlPanel/MEControlParameter.h"
#include "controlPanel/MEControlParameterLine.h"

QString toString(double value)
{
    return QString::number(value, 'g', 9);
}

QString toString(const QVariant &value)
{
    return toString(value.toDouble());
}

/*!
    \class MEParameterPort
    \brief Base class for all kinds of parameter ports
*/

//!
//! constructor used by COVISE
//!
MEParameterPort::MEParameterPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 const QString &paramtype,
                                 const QString &description)
    : MEPort(node, scene, portname, description, MEPort::PARAM)
    , mapped(false)
    , sensitive(true)
    , appearanceType(NOTMAPPED)
    , parameterType(paramtype)
    , secondLine(NULL)
    , left1(NULL)
    , left2(NULL)
    , right1(NULL)
    , right2(NULL)
    , stopp(NULL)
    , timer(NULL)
    , controlLine(NULL)
    , moduleLine(NULL)
{
    porttype = PIN;
}

//!
//! constructor used by YAC
//!
MEParameterPort::MEParameterPort(MENode *node, QGraphicsScene *scene,
                                 const QString &portname,
                                 int paramtype,
                                 const QString &description,
                                 int ptype)
    : MEPort(node, scene, portname, description, MEPort::PARAM)
    , mapped(false)
    , sensitive(true)
    , partype(paramtype)
    , appearanceType(NOTMAPPED)
    , porttype(ptype)
    , secondLine(NULL)
    , left1(NULL)
    , left2(NULL)
    , right1(NULL)
    , right2(NULL)
    , stopp(NULL)
    , timer(NULL)
    , controlLine(NULL)
    , moduleLine(NULL)

{
    // add port in list

    if (porttype == PIN)
        node->pinlist << this;
    else
        node->poutlist << this;

    // set the right color

    portcolor = definePortColor();
    setHelpText();
}

MEParameterPort::~MEParameterPort()
{
}

//!
//! show/hide the control line if port was mapped/unmapped
//!
void MEParameterPort::setMapped(bool state)
{
    mapped = state;
    showControlLine();
}

//!
//! generate a tooltip for the parameter port
//!
void MEParameterPort::setHelpText()
{
    QString help = "<i><b>" + portname + "</i></b>" + "(" + description + ")" + "<br>" + getParamTypeString() + "<br>";
    setToolTip(help);
}

//!
//! set a new appearance type
//!
void MEParameterPort::setAppearance(int atype)
{

#ifndef YAC
    atype = qAbs(atype);
    atype = qMax(0, atype);
#endif

    // clear old appearance if exist

    if (qAbs(appearanceType) != atype)
        removeFromControlPanel();

    // set new appearance

    appearanceType = atype;
}

//!
//! show/hide a parameter port inside a module icon (YAC)
//!
void MEParameterPort::setShown(bool flag)
{
    setVisible(flag);
    shown = flag;

    if (moduleLine)
        moduleLine->changeLightPixmap(shown);

#ifdef YAC
    node->layoutItem();
#endif
}

//!
//! add/remove a parameter to/from the controlpanel
//!
void MEParameterPort::showControlLine()
{

    if (mapped)
        addToControlPanel();

    else
        removeFromControlPanel();

    if (moduleLine)
        moduleLine->changeMappedPixmap(mapped);
}

//!
//! map parameter to control panel
//!
void MEParameterPort::addToControlPanel()
{
    // create a control parameter window for the node
    if (!node->getControlInfo())
        node->createControlPanelInfo();

    // create a control parameter line for this port
    if (controlLine == NULL)
    {
        QWidget *w = node->getControlInfo()->getContainer();
        controlLine = new MEControlParameterLine(w, this);
    }

    node->getControlInfo()->insertParameter(controlLine);
}

//!
//! unmap parameter from control panel
//!
void MEParameterPort::removeFromControlPanel()
{
    if (controlLine != NULL)
    {
        node->getControlInfo()->removeParameter(controlLine);
        controlLine = NULL;
    }
}

//!
//! create the parameter line in the module parameter window
//!
void MEParameterPort::createParameterLine(QFrame *textFrame, QWidget *contentFrame)
{
    moduleLine = new MEModuleParameterLine(this, textFrame, contentFrame);
}

//!
//! enable/disable a parameter
//!
void MEParameterPort::setSensitive(bool flag)
{
    sensitive = flag;

    // control info

    if (controlLine)
        controlLine->setEnabled(sensitive);

    //module info table

    if (node->getModuleInfo())
        moduleLine->setEnabled(sensitive);
}

//!
//! look if parameter has a control panel icon
//!
bool MEParameterPort::hasControlLine()
{
    if (controlLine == NULL)
        return false;
    else
        return true;
}

//!
//! get a color depending on the port type (YAC)
//!
QColor MEParameterPort::definePortColor()
{
    QColor col;
    col = Qt::black;

#ifdef YAC

    switch (porttype)
    {
    case PIN:
    case POUT:
        col = MEMainHandler::s_paramColor;
        break;
    }
#endif

    setBrush(col);
    return (col);
}

//!
//! send a PARAM message to controller
//!
void MEParameterPort::sendParamMessage(const QString &value)
{
#ifdef YAC

    Q_UNUSED(value);

#else

    MEMainHandler::instance()->mapWasChanged("PARAM");

    QStringList buffer;
    buffer << "PARAM" << node->getName() << node->getNumber() << node->getHostname()
           << portname << parameterType << value;

    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);

    buffer.clear();
    sendExecuteMessage();
#endif
}

//!
//! get the right parameter type
//!
int MEParameterPort::getParamType()
{

#ifdef YAC

    return partype;

#else

    if (parameterType == "IntSlider")
        return T_INTSLIDER;

    else if (parameterType == "FloatSlider")
        return T_FLOATSLIDER;

    else if (parameterType == "IntScalar")
        return T_INT;

    else if (parameterType == "FloatScalar")
        return T_FLOAT;

    else if (parameterType == "Browser")
        return T_BROWSER;

    else if (parameterType == "IntVector")
        return T_INTVECTOR;

    else if (parameterType == "FloatVector")
        return T_FLOATVECTOR;

    else if (parameterType == "Boolean")
        return T_BOOLEAN;

    else if (parameterType == "Choice")
        return T_CHOICE;

    else if (parameterType == "String")
        return T_STRING;

    else if (parameterType == "Timer")
        return T_TIMER;

    else if (parameterType == "Colormap")
        return T_COLORMAP;

    else if (parameterType == "Color")
        return T_COLOR;

    return 0;
#endif
}

//!
//! get the parameter type as a string
//!
QString MEParameterPort::getParamTypeString()
{

#ifdef YAC
    switch (partype)
    {
    case T_INTSLIDER:
        return QString("IntSlider");
    case T_FLOATSLIDER:
        return QString("FloatSlider");
    case T_INT:
        return QString("Integer");
    case T_FLOAT:
        return QString("Scalar");
    case T_BROWSER:
        return QString("Browser");
    case T_FLOATVECTOR:
        return QString("Vector");
    case T_INTVECTOR:
        return QString("Integer Vector");
    case T_BOOLEAN:
        return QString("Boolean");
    case T_CHOICE:
        return QString("Choice");
    case T_STRING:
        return QString("String");
    case T_TIMER:
        return QString("Timer");
    }

    return QString("Unknown");

#else

    return parameterType;
#endif
}

//!
//! define a parameter
//!
void MEParameterPort::defineParam(QString value, int apptype)
{
#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    Q_UNUSED(value);

    // set appearanceType and map mode
    if (apptype > 0)
    {
        setAppearance(apptype);
        mapped = true;
    }

    if (apptype == 0)
    {
        setAppearance(1);
        mapped = true;
    }

    // update control window
    if (mapped)
        showControlLine();

    hide();
#endif
}

#ifdef YAC

//!
//! set parameter values (YAC)
//!
void MEParameterPort::setValues(covise::coRecvBuffer &tb)
{
    setValues(tb);
}
#endif

//!
//! set some variables (YAC)
//!
void MEParameterPort::addItems(bool shown, bool lmapped, int lappearanceType)
{
    mapped = lmapped;
    setVisible(shown);
    appearanceType = lappearanceType;
}

//!
//! send an EXECUTE if status is ExecOnChange or a timer is active
//!
void MEParameterPort::sendExecuteMessage()
{
    if (MEMainHandler::instance()->isExecOnChange() || (timer && timer->isActive())) {
        node->sendExec();
    }
}

//!
//! send a ADD/RM_PANEL message to controller
//!
void MEParameterPort::sendPanelMessage(const QString &key)
{
#ifdef YAC

    Q_UNUSED(key);

#else

    QStringList buff;
    buff << key;
    buff << node->getName() << node->getNumber() << node->getHostname();
    buff << portname;

    if (key.contains("ADD"))
    {
        if (appearanceType == NOTMAPPED) // default
            appearanceType = NORMAL;
        else
            appearanceType = qAbs(appearanceType); // only unmap current appearance
    }

    else if (key.contains("RM"))
    {
        appearanceType = -appearanceType; // only unmap current appearance
    }

    buff << QString::number(appearanceType);
    QString data = buff.join("\n");

    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buff.clear();
#endif
}

//!
//! colorize parameter text when line has focus
//!
void MEParameterPort::setFocusCB(bool state)
{
    if (moduleLine)
        moduleLine->colorTextFrame(state);

    if (controlLine)
        controlLine->colorTextFrame(state);
}

//!
//! the appearance popup menu was clicked
//!
void MEParameterPort::appearanceCB(const QString &tmp)
{

#ifdef YAC
    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;

    // set new appearance type
    if (QString::compare(tmp, "Integer") == 0 || QString::compare(tmp, "Float") == 0 || QString::compare(tmp, "String") == 0)
        sb << A_STRING;

    else if (QString::compare(tmp, "Slider") == 0)
        sb << A_SLIDER;

    else if (QString::compare(tmp, "Player") == 0)
        sb << A_STEPPER;

    else if (QString::compare(tmp, "Spinbox") == 0)
        sb << A_SPINBOX;

    else if (QString::compare(tmp, "Dial") == 0)
        sb << A_DIAL;

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_CHANGE_APPEARANCE, sb);
    MEMainHandler::instance()->mapWasChanged("cmapCB");

#else

    // set new appearance type
    int appearanceType = 0;
    if (tmp == "Integer" || tmp == "Float")
        appearanceType = 1;

    else if (tmp == "Slider")
        appearanceType = 1;

    else if (tmp == "Player" || tmp == "Stepper")
        appearanceType = 2;

    // send message to controller
    if (appearanceType != 0)
    {
        QStringList list;
        list << "APP_CHANGE";
        list << node->getName() << node->getNumber() << node->getHostname();
        list << portname << QString::number(appearanceType);
        QString data = list.join("\n");

        MEMainHandler::instance()->mapWasChanged("MEParameterPort::appCB");
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
        list.clear();
    }
#endif
}

// these are callbacks from player or stepper widgets

//!
//! PlayerCB
//!
void MEParameterPort::left1CB()
{
    // reset timer variables
    if (timer)
    {
        timer->setAction(METimer::BACKWARD);
        timer->setActive(true);
    }

    // set a new value
    minusNewValue();
}

//!
//! PlayerCB
//!
void MEParameterPort::right1CB()
{
    // reset timer variables
    if (timer)
    {
        timer->setAction(METimer::FORWARD);
        timer->setActive(true);
    }

    // set a new value
    plusNewValue();
}

//!
//! PlayerCB
//!
void MEParameterPort::reverseCB()
{
    // create a timer record
    if (!timer)
    {
        timer = new METimer(this);
        MEMainHandler::instance()->timerList.append(timer);
    }

    // reset timer variables
    timer->setAction(METimer::REVERSE);
    timer->setActive(true);

    // enable the buttons
    stopp->setEnabled(true);
    left1->setEnabled(false);
    left2->setEnabled(false);
    right1->setEnabled(false);
    right2->setEnabled(false);

    // set a new value
    minusNewValue();
}

//!
//! PlayerCB
//!
void MEParameterPort::playCB()
{
    // create a timer record
    if (!timer)
    {
        timer = new METimer(this);
        MEMainHandler::instance()->timerList.append(timer);
    }

    // reset timer variables
    timer->setAction(METimer::PLAY);
    timer->setActive(true);

    // enable the buttons
    stopp->setEnabled(true);
    left1->setEnabled(false);
    left2->setEnabled(false);
    right1->setEnabled(false);
    right2->setEnabled(false);

    // set a new value
    plusNewValue();
}

//!
//! PlayerCB
//!
void MEParameterPort::stopCB()
{
    // reset timer variables
    if (timer)
    {
        timer->setAction(METimer::STOP);
        timer->setActive(false);
    }

    // enable all buttons
    stopp->setEnabled(true);
    left1->setEnabled(true);
    left2->setEnabled(true);
    right1->setEnabled(true);
    right2->setEnabled(true);
}

void MEParameterPort::plusNewValue() {}
void MEParameterPort::minusNewValue() {}
