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
#ifndef PROJECTIONSETTINGS_HPP
#define PROJECTIONSETTINGS_HPP

#include <QDialog>

#include <proj_api.h>

namespace Ui
{
class ProjectionSettings;
}

class ProjectionSettings : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectionSettings();
    virtual ~ProjectionSettings();

    void transform(double &x, double &y, double &z);
    projPJ pj_from, pj_to;
    double XOffset;
    double YOffset;
    double ZOffset;
    static ProjectionSettings *instance()
    {
        return inst;
    };

private:
    static ProjectionSettings *inst;
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ProjectionSettings *ui;
};

#endif // OSMIMPORT_HPP
