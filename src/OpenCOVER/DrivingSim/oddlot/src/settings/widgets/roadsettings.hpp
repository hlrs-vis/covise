/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#ifndef ROADSETTINGS_HPP
#define ROADSETTINGS_HPP

#include "src/settings/settingselement.hpp"
#include "src/data/prototypemanager.hpp"

namespace Ui
{
class RoadSettings;
}

class RSystemElementRoad;

class RoadSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, RSystemElementRoad *road);
    virtual ~RoadSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateProperties();
    void updateRoadLinks();
    void addLaneSectionPrototypes();
    //	void						updateSectionCount();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_editingFinished();
    void on_newButton_released();
    void on_addButton_released();
    void on_laneSectionComboBox_activated(int);
    void on_roadTypeComboBox_activated(int);
    void on_elevationComboBox_activated(int);
    void on_superelevationComboBox_activated(int);
    void on_crossfallComboBox_activated(int);

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::RoadSettings *ui;

    PrototypeManager *prototypeManager_;

    RSystemElementRoad *road_;

    bool init_;
};

#endif // ROADSETTINGS_HPP
