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
#ifndef OSCSETTINGS_HPP
#define OSCSETTINGS_HPP

#include <QDialog>

namespace Ui
{
class OSCSettings;
}

class OSCSettings : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCSettings();
    virtual ~OSCSettings();

    bool readValidation();
    bool loadDefaults();
    
    static OSCSettings *instance()
    {
        return inst;
    };

private:
    static OSCSettings *inst;
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();

	signals:
	void readValidationChanged(bool);

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::OSCSettings *ui;

	bool validation;
};

#endif // OSCSETTINGS_HPP
