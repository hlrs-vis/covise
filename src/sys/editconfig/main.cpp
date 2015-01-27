/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/editor/coConfigEditor.h>
#include <qapplication.h>
#include <q3mainwindow.h>

#include <config/coConfig.h>

int main(int argc, char **argv)
{

    QApplication app(argc, argv);
    app.setStyle(coConfig::getInstance()->getValue("style", "UICONFIG.QTSTYLE").lower());

    Q3MainWindow *mainWindow = new Q3MainWindow();
    coConfigEditor *editor = new coConfigEditor(mainWindow);
    mainWindow->setCaption("co Configuration Editor");
    mainWindow->setCentralWidget(editor);
    mainWindow->resize(800, 600);
    editor->plug(mainWindow->topDock());

    app.setMainWidget(mainWindow);

    mainWindow->show();

    return app.exec();
}
