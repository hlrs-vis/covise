/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/29/2010
**
**************************************************************************/

#ifndef LANEROADMARKSETTINGS_HPP
#define LANEROADMARKSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class LaneRoadMarkSettings;
}

class LaneRoadMarkSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneRoadMarkSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, LaneRoadMark *laneRoadMark);
    virtual ~LaneRoadMarkSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateSOffset();
    void updateType();
    void updateWeight();
    void updateColor();
    void updateWidth();
    void updateLaneChange();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_offsetBox_editingFinished();
    void on_typeBox_currentIndexChanged(const QString &text);
    void on_weightBox_currentIndexChanged(const QString &text);
    void on_colorBox_currentIndexChanged(const QString &text);
    void on_widthBox_editingFinished();
    void on_laneChangeBox_currentIndexChanged(const QString &text);

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::LaneRoadMarkSettings *ui;

    LaneRoadMark *roadMark_;

    bool init_;
};

#endif // LANEROADMARKSETTINGS_HPP
