/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// MainWindow.cpp

#include "MainWindow.h"
#include "Debug.h"

#include <QMenuBar>
#include <QGLFormat>
#include <QMessageBox>

#include <stdio.h>
#include <string>

MainWindow::MainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , canClose(true)
{
    ui.setupUi(this);

    //
    // Create main widget
    //

    QGLFormat format;

    format.setDirectRendering(true);
    format.setRgba(true);
    format.setDoubleBuffer(true);
    format.setSwapInterval(0);
    format.setStencil(false);
    format.setDepth(false);

    mainWidget = new MainWidget(format, this);

    connect(mainWidget,
            SIGNAL(signalMessage(int /*MainWidget::Message*/, int, int, int, int)),
            this,
            SLOT(message(int /*MainWidget::Message*/, int, int, int, int)));

    setCentralWidget(mainWidget);

    //
    // Adjust actions
    //

    connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(ui.actionFullscreen, SIGNAL(triggered()), this, SLOT(toggleFullscreen()));
    connect(ui.actionConnect, SIGNAL(triggered()), this, SLOT(connectToHost()));
    connect(ui.actionDisconnect, SIGNAL(triggered()), this, SLOT(disconnectFromHost()));

    //
    // Create dialog
    //

    networkDialog = new NetworkDialog(this);
}

MainWindow::~MainWindow()
{
    delete networkDialog;
    delete mainWidget;
}

void MainWindow::closeEvent(QCloseEvent *closeEvent)
{
    if (canClose)
        closeEvent->accept();
    else
        closeEvent->ignore();
}

void MainWindow::message(int type, int x, int y, int z, int /*w*/)
{
    ASSERT(dynamic_cast<MainWidget *>(sender())); // Handles messages from MainWidget only!

    switch (type)
    {
    case MainWidget::Message_Quit:
    {
        close();
    }
    break;

    case MainWidget::Message_ToggleFullscreen:
    {
        toggleFullscreen();
    }
    break;

    case MainWidget::Message_FPS:
    {
        setWindowTitle(QString("Comote (%1x%2) [%3 fps]").arg(x).arg(y).arg(z));
    }
    break;

    case MainWidget::Message_ServerConnecting:
    {
        canClose = false;

        ui.actionExit->setEnabled(false);
        ui.actionConnect->setEnabled(false);
        ui.actionDisconnect->setEnabled(false);
    }
    break;

    case MainWidget::Message_ServerConnected:
    {
        canClose = true;

        ui.actionExit->setEnabled(true);
        ui.actionConnect->setEnabled(false);
        ui.actionDisconnect->setEnabled(true);
    }
    break;

    case MainWidget::Message_ServerDisconnected:
    case MainWidget::Message_ServerFailed:
    {
        if (type == MainWidget::Message_ServerFailed)
        {
            QMessageBox::critical(this, "Comote", "Failed to connect");
        }

        canClose = true;

        ui.actionExit->setEnabled(true);
        ui.actionConnect->setEnabled(true);
        ui.actionDisconnect->setEnabled(false);
    }
    break;

    default:
        break;
    }
}

void MainWindow::toggleFullscreen()
{
    if (isFullScreen())
    {
        showNormal();
        //if (menuBar()) menuBar()->show();
    }
    else
    {
        showFullScreen();
        //if (menuBar()) menuBar()->hide();
    }
}

void MainWindow::connectToHost()
{
    if (QDialog::Accepted == networkDialog->exec())
    {
        if (!mainWidget->connectToHost(networkDialog->getHostname(), networkDialog->getTCPPort(), networkDialog->getUDPPort()))
        {
        }
    }
}

void MainWindow::disconnectFromHost()
{
    if (!mainWidget->disconnectFromHost())
    {
    }
}
