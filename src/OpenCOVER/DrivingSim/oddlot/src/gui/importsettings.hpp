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
#ifndef IMPORTSETTINGS_HPP
#define IMPORTSETTINGS_HPP

#include <QDialog>

namespace Ui
{
class ImportSettings;
}

class ImportSettings : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ImportSettings();
    virtual ~ImportSettings();

    double LinearError();
    double CurveError();
    bool importPrimary();
    bool importSecondary();
    bool importTertiary();
    bool importMotorway();
    bool importService();
    bool importPath();
    bool importSteps();
    bool importTrack();
    bool importFootway();
    bool importResidential();
    bool importLiving_street();
    bool importCycleway();
    bool importTurning_circle();
    bool importPedestrian();
    bool importUnclassified();
    bool maximizeCurveRadius();
    static ImportSettings *instance()
    {
        return inst;
    };

private:
    static ImportSettings *inst;
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ImportSettings *ui;
};

#endif // OSMIMPORT_HPP
