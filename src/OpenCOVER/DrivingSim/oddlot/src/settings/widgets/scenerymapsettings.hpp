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

#ifndef SCENERYMAPSETTINGS_HPP
#define SCENERYMAPSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class SceneryMapSettings;
}

class SceneryMapSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SceneryMapSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, SceneryMap *sceneryMap);
    virtual ~SceneryMapSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updatePosition();
    void updateSize();
    void updateOpacity();
    void updateFilename();
    void updateDataFilename();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_changeFileButton_released();
    void on_changeDataButton_released();
    void on_xBox_editingFinished();
    void on_yBox_editingFinished();
    void on_widthBox_editingFinished();
    void on_heightBox_editingFinished();
    void on_opacityBox_editingFinished();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::SceneryMapSettings *ui;

    SceneryMap *sceneryMap_;
    Heightmap *heightmap_;
};

#endif // SCENERYMAPSETTINGS_HPP
