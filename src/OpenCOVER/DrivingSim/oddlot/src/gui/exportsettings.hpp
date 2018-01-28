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
#ifndef ExportSETTINGS_HPP
#define ExportSETTINGS_HPP

#include <QDialog>

namespace Ui
{
class ExportSettings;
}

class ExportSettings : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:

	enum ExportIDVariants
	{
		EXPORT_ORIGINAL=0,
		EXPORT_NUMERICAL,
		EXPORT_TILE_ID
	};
    explicit ExportSettings();
    virtual ~ExportSettings();

	ExportSettings::ExportIDVariants ExportSettings::ExportIDVariant();

    static ExportSettings *instance()
    {
        return inst;
    };

private:
    static ExportSettings *inst;
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ExportSettings *ui;
};

#endif // OSMExport_HPP
