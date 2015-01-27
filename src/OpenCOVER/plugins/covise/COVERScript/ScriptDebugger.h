/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCRIPTDEBUGGER_H
#define SCRIPTDEBUGGER_H

#include <QObject>

class QAction;
class QMenu;

class QScriptEngineDebugger;

class ScriptEngineProvider;

class ScriptDebugger : public QObject
{
    Q_OBJECT

public:
    ScriptDebugger(ScriptEngineProvider *plugin);

public slots:
    void show();
    void hide();

    void loadScript(const QString &filename);

private slots:
    void load();

private:
    QScriptEngineDebugger *debugger;
    ScriptEngineProvider *provider;

    QMenu *fileMenu;
    QAction *loadAction;
};

#endif // SCRIPTDEBUGGER_H
