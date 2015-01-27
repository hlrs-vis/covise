/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QHBoxLayout>

#include "MEStringPort.h"
#include "MELineEdit.h"
#include "MEMessageHandler.h"
#include "widgets/MEUserInterface.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"

/*****************************************************************************
 *
 * Class MEStringPort
 * normal string input
 *
 *****************************************************************************/

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEStringPort::MEStringPort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           const QString &paramtype,
                           const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
{
    editLine[MODULE] = editLine[CONTROL] = NULL;
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEStringPort::MEStringPort(MENode *node, QGraphicsScene *scene,
                           const QString &portname,
                           int paramtype,
                           const QString &description,
                           int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
{

    editLine[MODULE] = editLine[CONTROL] = NULL;
}

//------------------------------------------------------------------------
MEStringPort::~MEStringPort()
//------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------
// restore saved parameters
// after the user has pressed cancel in module parameter window
//------------------------------------------------------------------------
void MEStringPort::restoreParam()
{
    svalue = svalueold;
    sendParamMessage();
}

//------------------------------------------------------------------------
// save current value for further use
//------------------------------------------------------------------------
void MEStringPort::storeParam()
{
    svalueold = svalue;
}

//------------------------------------------------------------------------
// module has requested parameter
//------------------------------------------------------------------------
void MEStringPort::moduleParameterRequest()
{
    sendParamMessage();
}

//------------------------------------------------------------------------
// define one parameter of a module,	init from controller
//------------------------------------------------------------------------
void MEStringPort::defineParam(QString value, int apptype)
{
#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else
    svalue = value.replace(QChar('\177'), " ");
    if (svalue.size() == 1 && svalue[0] == '\001')
        svalue = "";
    MEParameterPort::defineParam(svalue, apptype);
#endif
}

//------------------------------------------------------------------------
// update parameter value comes from controller
//------------------------------------------------------------------------
void MEStringPort::modifyParam(QStringList list, int noOfValues, int istart)
{
#ifdef YAC

    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);

#else

    Q_UNUSED(noOfValues);

    if (istart >= list.count())
    {
        MEUserInterface::instance()->printMessage("ModifyParam of type string: no string given");
        return;
    }

    svalue = list[istart].replace(QChar('\177'), " ");
    if (svalue.size() == 1 && svalue[0] == '\001')
        svalue = "";

    // modify module & control line content
    if (editLine[MODULE])
        editLine[MODULE]->setText(svalue);

    if (editLine[CONTROL])
        editLine[CONTROL]->setText(svalue);
#endif
}

//------------------------------------------------------------------------
// update parameter value comes from controller
//------------------------------------------------------------------------
void MEStringPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    svalue = lvalue.replace(QChar('\177'), " ");
    if (svalue.size() == 1 && svalue[0] == '\001')
        svalue = "";

    // modify module & control line content
    if (editLine[MODULE])
        editLine[MODULE]->setText(svalue);

    if (editLine[CONTROL])
        editLine[CONTROL]->setText(svalue);
#endif
}

//------------------------------------------------------------------------
// send a PARAM message to controller
//------------------------------------------------------------------------
void MEStringPort::sendParamMessage()
{
    QChar sep = '\177';
    QString tmp = svalue.replace(" ", sep);
    if (tmp == "")
        tmp = QChar('\001');
    MEParameterPort::sendParamMessage(tmp);
}

//------------------------------------------------------------------------
// make the layout for the module parameter widget
//------------------------------------------------------------------------
void MEStringPort::makeLayout(layoutType type, QWidget *w)
{

    QHBoxLayout *hBox = new QHBoxLayout(w);
    hBox->setMargin(2);
    hBox->setSpacing(2);

    editLine[type] = new MELineEdit();
    editLine[type]->setMinimumWidth(MEMainHandler::instance()->getSliderWidth());
    editLine[type]->setText(svalue);

    connect(editLine[type], SIGNAL(contentChanged(const QString &)), this, SLOT(textCB(const QString &)));
    connect(editLine[type], SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));

    hBox->addWidget(editLine[type], 1);
}

//------------------------------------------------------------------------
// unmap parameter from control panel
//------------------------------------------------------------------------
void MEStringPort::removeFromControlPanel()
{
    MEParameterPort::removeFromControlPanel();
    editLine[CONTROL] = NULL;
}

//------------------------------------------------------------------------
// get the current value
//------------------------------------------------------------------------
void MEStringPort::textCB(const QString &text)
{

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << text;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    svalue = text;
    sendParamMessage();
#endif

    // inform parent widget that value has been changed
    node->setModified(true);
}

#ifdef YAC

//------------------------------------------------------------------------
void MEStringPort::setValues(covise::coRecvBuffer &tb)
//------------------------------------------------------------------------
{
    const char *name;

    tb >> name;
    svalue = name;

    // modify module & control line content
    if (editLine[MODULE])
        editLine[MODULE]->setText(svalue);

    if (editLine[CONTROL])
        editLine[CONTROL]->setText(svalue);
}
#endif
