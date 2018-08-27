/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   03/2013
**
**************************************************************************/
#ifndef LODSETTINGS_HPP
#define LODSETTINGS_HPP

#include <QDialog>

namespace Ui
{
class LODSettings;
}

class LODSettings : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LODSettings();
    virtual ~LODSettings();

    double TopViewEditorPointsPerMeter;
    double HeightEditorPointsPerMeter;
    double SignalEditorScalingLevel;

    //int port;
    //QString hostname;

    static LODSettings *instance()
    {
        return inst;
    };
    /*bool doConnect();
    void setConnected(bool c);
    int getPort();*/

private:
    static LODSettings *inst;
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::LODSettings *ui;
};

#endif // OSMIMPORT_HPP
