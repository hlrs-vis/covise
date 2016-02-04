/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OpenGeoGroup.h"
#include "MainWindow.h"
#include "HelpFuncs.h"

#include <QFileDialog>

OpenGeoGroup::OpenGeoGroup(MainWindow *parent)
    : QGroupBox(parent)
    , mainWindow(parent)
{
    screen = dynamic_cast<ScreenLoad *>(Scene::getScreen());

    //Gui aufbauen
    ui.setupUi(this);
    connect(ui.toolButton, SIGNAL(clicked()), this, SLOT(toolButtonClicked()));

    //Gui-Elemente befüllen
    init();
}

OpenGeoGroup::~OpenGeoGroup()
{
}

void OpenGeoGroup::init()
{
    ui.lineEdit->setText(screen->getFilename().c_str());
}

void OpenGeoGroup::update()
{
    mainWindow->update();
}

bool OpenGeoGroup::toolButtonClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "./",
                                                    tr("ConfigFile (*.osg)"));

    //Benutzer abbruch
    if ((fileName.isEmpty()) || (!checkFile(fileName.toStdString())))
        return false;

    screen->setFilename(fileName.toStdString());
    update();
    return true;
}

bool OpenGeoGroup::checkFile(std::string file)
{
    if (screen->fileIsValid(file))
        return true;
    else
    {
        mainWindow->setupErrMsgBox("The selected Geometry-File is invalid!");
        return false;
    }
}
