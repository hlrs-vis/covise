/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/26/2010
**
**************************************************************************/

#ifndef TRACKCOMPONENTSETTINGS_HPP
#define TRACKCOMPONENTSETTINGS_HPP

#include "src/settings/settingselement.hpp"

namespace Ui
{
class TrackComponentSettings;
}

class TrackComponent;

class TrackComponentSettings : public SettingsElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackComponentSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, TrackComponent *trackComponent);
    virtual ~TrackComponentSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    void updateTransformation();
    void updateS();
    void updateLength();
    void updateCurvature();

    //################//
    // SLOTS          //
    //################//

private slots:
    void on_factorBox_editingFinished();
    void on_lengthBox_editingFinished();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::TrackComponentSettings *ui;

    TrackComponent *trackComponent_;

    TrackElementLine *line_;
    TrackElementArc *arc_;
    TrackElementSpiral *spiral_;
	TrackElementPoly3 *poly3_;
	TrackElementCubicCurve *c_curve_;

    TrackSpiralArcSpiral *sparcs_;
};

#endif // TRACKCOMPONENTSETTINGS_HPP
