/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScriptDebugger.h"

#include "ScriptEngineProvider.h"

#include <QAction>
#include <QFileDialog>
#include <QMenu>
#include <QMenuBar>
#include <QScriptEngineDebugger>
#include <QMainWindow>

#define engine this->provider->engine()

ScriptDebugger::ScriptDebugger(ScriptEngineProvider *provider)
    : QObject(0)
    , provider(provider)
    , fileMenu(0)
    , loadAction(0)
{

    this->debugger = new QScriptEngineDebugger();
    this->debugger->attachTo(&engine);

    QMainWindow *win = this->debugger->standardWindow();

    this->fileMenu = new QMenu(tr("File"), win);

    this->loadAction = this->fileMenu->addAction(tr("Open Script"), this, SLOT(load()));

    win->menuBar()->addMenu(this->fileMenu);
}

void ScriptDebugger::show()
{
    this->debugger->standardWindow()->show();
}

void ScriptDebugger::hide()
{
    this->debugger->standardWindow()->hide();
}

void ScriptDebugger::loadScript(const QString &filename)
{
    this->provider->loadScript(filename);
}

void ScriptDebugger::load()
{
    QString filename = QFileDialog::getOpenFileName(this->debugger->standardWindow(), tr("Open Script"), ".", "Scripts (*.js *.qs);;All Files (*.*)");

    if (filename != QString::null)
        loadScript(filename);
}
