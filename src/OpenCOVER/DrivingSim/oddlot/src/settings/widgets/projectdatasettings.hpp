/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/25/2010
**
**************************************************************************/

#ifndef PROJECTDATASETTINGS_HPP
#define PROJECTDATASETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class ProjectDataSettings;
}

class ProjectDataSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectDataSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, ProjectData *projectData);
    virtual ~ProjectDataSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateName();
    void updateVersion();
    void updateDate();
    void updateSize();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_nameButton_released();
    void on_autoSizeMeButton_released();
    void on_versionBox_editingFinished();

    void on_northBox_editingFinished();
    void on_southBox_editingFinished();
    void on_eastBox_editingFinished();
    void on_westBox_editingFinished();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ProjectDataSettings *ui;

    ProjectData *projectData_;
};

#endif // PROJECTDATASETTINGS_HPP
