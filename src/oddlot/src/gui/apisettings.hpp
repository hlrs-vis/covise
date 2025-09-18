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
#ifndef APISETTINGS_HPP
#define APISETTINGS_HPP

#include <QDialog>

namespace Ui
{
    class APISettings;
}

class MainWindow;

class APISettings : public QDialog
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit APISettings(MainWindow* mw);
    virtual ~APISettings();

    QString GoogleAPIKey;
    QString BINGAPIKey;

    //int port;
    //QString hostname;

    static APISettings *instance()
    {
        return inst;
    };
    MainWindow* mainWindow;

private:
    static APISettings *inst;
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::APISettings *ui;
};

#endif // OSMIMPORT_HPP
