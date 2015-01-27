/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/19/2010
**
**************************************************************************/

#ifndef ELEVATIONSETTINGS_HPP
#define ELEVATIONSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class ElevationSettings;
}

class ElevationMoveHandle;
class ElevationEditor;

class ElevationSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, ElevationSection *elevationSection);
    virtual ~ElevationSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();
    void updateHeight();
    void enableElevationSettingParams(bool);
    ElevationMoveHandle *getFirstSelectedMoveHandle();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_heightBox_editingFinished();
    void on_sSpinBox_editingFinished();
    void on_slopeSpinBox_editingFinished();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ElevationSettings *ui;

    ElevationSection *elevationSection_;
};

#endif // ELEVATIONSETTINGS_HPP
