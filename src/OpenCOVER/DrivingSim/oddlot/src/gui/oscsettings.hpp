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
    explicit OSCSettings(const QString &dir);
    virtual ~OSCSettings();

    bool readValidation();
    bool loadDefaults();
	QString getCatalogDir();
    
    static OSCSettings *instance()
    {
        return inst;
    };

private:
    static OSCSettings *inst;

//################//
// SIGNALS        //
//################//

signals:
    void directoryChanged();
	void readValidationChanged(bool);

    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();
	void dirPushButtonPressed();


    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::OSCSettings *ui;

	bool validation_;
	QString catalogDir_;
};

#endif // OSCSETTINGS_HPP
