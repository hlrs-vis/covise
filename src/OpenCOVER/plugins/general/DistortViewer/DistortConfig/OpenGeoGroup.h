/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENGEOGROUP_H
#define OPENGEOGROUP_H

#include <QGroupBox>
#include "ui_OpenGeoGroup.h"
#include "ScreenLoad.h"
#include "Scene.h"

class MainWindow;

class OpenGeoGroup : public QGroupBox
{
    Q_OBJECT

public:
    OpenGeoGroup(MainWindow *parent = 0);
    ~OpenGeoGroup();
    void update();
    void init();
    bool checkFile(std::string file);

public slots:
    bool toolButtonClicked();

private:
    Ui::OpenGeoGroup ui;
    MainWindow *mainWindow;
    ScreenLoad *screen;
};

#endif // OPENGEOGROUP_H
