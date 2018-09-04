/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/23/2010
**
**************************************************************************/

#ifndef LANESETTINGS_HPP
#define LANESETTINGS_HPP

#include "src/settings/settingselement.hpp"
#include "src/graph/profilegraph.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class LaneEditor;
class SectionHandle;
class BaseLaneMoveHandle;

class RoadSystemItem;
namespace Ui
{
class LaneSettings;
}

class LaneSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Lane *lane);
    virtual ~LaneSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

    SectionHandle *getSectionHandle()
    {
        return insertWidthSectionHandle_;
    };
    Lane *getLane()
    {
        return lane_;
    };

private:
    void updateId();
    void updateType();
    void updateLevel();
    void updatePredecessor();
    void updateSuccessor();
    void updateWidth();
	BaseLaneMoveHandle * getFirstSelectedLaneWidthHandle();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_typeBox_currentIndexChanged(const QString &text);
    void on_levelBox_stateChanged(int state);
    void on_predecessorCheckBox_stateChanged(int state);
    void on_predecessorBox_valueChanged(int i);
    void on_successorCheckBox_stateChanged(int state);
    void on_successorBox_valueChanged(int i);
    void on_addButton_released();
    void on_addWidthButton_released();
    void on_widthSpinBox_valueChanged(double w);
	void activateInsertGroupBox(bool);
	void activateWidthGroupBox(bool);

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::LaneSettings *ui;

    Lane *lane_;

    bool init_;

    LaneEditor *laneEditor_;
    SectionHandle *insertWidthSectionHandle_;
};

#endif // LANESETTINGS_HPP
