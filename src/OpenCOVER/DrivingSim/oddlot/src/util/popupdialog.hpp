/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   10/8/2010
 **
 **************************************************************************/

#ifndef POPUPDIALOG_HPP
#define POPUPDIALOG_HPP

#include <QDialog>

class PopUpDialog : public QDialog
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit PopUpDialog(QWidget *parent = 0) :QDialog(parent) {};

    //################//
    // EVENTS         //
    //################//

protected:
    virtual void closeEvent(QCloseEvent *event);
};

#endif // POPUPDIALOG_HPP
