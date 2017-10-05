/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QMenu>
#include <QContextMenuEvent>

#include "MEParameterAppearance.h"
#include "MEParameterPort.h"
#include "handler/MEMainHandler.h"

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------

MEParameterAppearance::MEParameterAppearance(QWidget *parent, MEParameterPort *p)
    : QLabel(parent)
    , appPopup(NULL)
    , port(p)
{

#ifdef YAC
    il_scalar << "Integer"
              << "Slider"
              << "Player"
              << "Spinbox";
    fl_scalar << "Float"
              << "Slider"
              << "Player"
              << "Spinbox";
#else
    il_scalar << "Integer"
              << "Stepper";
    fl_scalar << "Float"
              << "Stepper";
    ll_slider << "Slider"
              << "Player";
#endif

    setFont(MEMainHandler::s_boldFont);

    // display parameter name
    setText(port->getName());

    // disable focis policy
    setFocusPolicy(Qt::NoFocus);

    // set a tooltip
    QString tipText("<b>Type: </b>" + port->getParamTypeString() + "<br><i>" + port->getDescription() + "</i>");
    setToolTip(tipText);
}

//------------------------------------------------------------------------
// store available text
//------------------------------------------------------------------------
void MEParameterAppearance::insertText(QStringList list)
{
    appPopup->addAction("Possible Appearance Types");
    appPopup->addSeparator();

    for ( int i = 0; i < list.count(); i++)
    {
        QAction *ac = appPopup->addAction(list[i]);
        appList << ac;
        connect(ac, SIGNAL(triggered()), this, SLOT(appearanceCB()));
    }
}

//------------------------------------------------------------------------
// user has selected an appearance type
//------------------------------------------------------------------------
void MEParameterAppearance::appearanceCB()
{
    // object that sent the signal
    const QObject *obj = sender();
    QAction *ac = (QAction *)obj;

    // find position in list
    int id = appList.indexOf(ac);

    QString tmp;
    switch (port->getParamType())
    {
    case MEParameterPort::T_FLOAT:
        tmp = fl_scalar[id];
        port->appearanceCB(tmp);
        break;

    case MEParameterPort::T_INT:
        tmp = il_scalar[id];
        port->appearanceCB(tmp);
        break;

    case MEParameterPort::T_INTSLIDER:
    case MEParameterPort::T_FLOATSLIDER:
        tmp = ll_slider[id];
        port->appearanceCB(tmp);
        break;
    }
}

//------------------------------------------------------------------------
// mouse pressed
//
// show a list of appearance types
//------------------------------------------------------------------------
void MEParameterAppearance::contextMenuEvent(QContextMenuEvent *e)
{

    // text for appearance combo box
    if (!appPopup)
    {
        appPopup = new QMenu(this);
        appPopup->setFont(MEMainHandler::s_normalFont);

        switch (port->getParamType())
        {
        case MEParameterPort::T_FLOAT:
            insertText(fl_scalar);
            break;

        case MEParameterPort::T_INT:
            insertText(il_scalar);
            break;

        case MEParameterPort::T_INTSLIDER:
        case MEParameterPort::T_FLOATSLIDER:
            insertText(ll_slider);
            break;

        default:
            appPopup->addAction("No Appearance Types available");
            break;
        }
    }

    appPopup->popup(e->globalPos());
}
