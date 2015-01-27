/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIUITab.h"

#include <sstream>
#include <iostream>
#include <QLayout>
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

TUIUITab::TUIUITab(int id, int type, QWidget *w, int parent, QString name)
    : TUITab(id, type, w, parent, name)
    , uiWidget(0)
{
    std::cerr << "TUIUITab::<init> info: creating..." << std::endl;
}

TUIUITab::~TUIUITab()
{
    std::cerr << "TUIUITab::<dest> info: destroying..." << std::endl;
}

char *TUIUITab::getClassName()
{
    return (char *)"TUIUITab";
}

void TUIUITab::setValue(int type, covise::TokenBuffer &tb)
{
    if (type == TABLET_UI_USE_DESCRIPTION)
    {
        int size;
        ushort *data;
        tb >> size;
        data = (ushort *)tb.getBinary(size);

        this->uiDescription = QString::fromUtf16(data);

        delete this->uiWidget;
        this->uiWidget = new TUIUIWidget(this->uiDescription, this);
        this->widget->layout()->addWidget(this->uiWidget);

        connect(this->uiWidget, SIGNAL(command(QString, QString)), this, SLOT(sendCommand(QString, QString)));

        //std::cerr << "TUIUITab::setValue info: ui " << std::endl << qPrintable(this->uiDescription) << std::endl;
    }
    else if (type == TABLET_UI_COMMAND)
    {
        std::string target;
        tb >> target;

        int size;
        ushort *data;
        tb >> size;
        data = (ushort *)tb.getBinary(size);

        QString command = QString::fromUtf16(data);

        //std::cerr << "TUIUITab::setValue info: command = " << qPrintable(command) << std::endl;
        this->uiWidget->processMessage(command);
    }

    TUITab::setValue(type, tb);
}

void TUIUITab::sendCommand(const QString &target, const QString &command)
{
    //std::cerr << "TUIUITab::sendCommand info: sending command (" <<
    //             qPrintable(target) << ") " << qPrintable(command) << std::endl;

    covise::TokenBuffer tb;
    unsigned long commandSize = (unsigned long)(command.size() + 1) * 2;

    tb << ID;
    tb << TABLET_UI_COMMAND;
    tb << target.toLocal8Bit().data();
    tb << (uint64_t)commandSize;
    tb.addBinary((const char *)command.utf16(), (command.size() + 1) * 2);
    TUIMainWindow::getInstance()->send(tb);
}
