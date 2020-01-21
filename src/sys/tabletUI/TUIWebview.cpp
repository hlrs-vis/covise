/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIWebview.h"
#include "TUIApplication.h"
#include <stdio.h>
#include <qcheckbox.h>
#include <qradiobutton.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qpixmap.h>
#include <net/tokenbuffer.h>

/// Constructor
TUIWebview::TUIWebview(int id, int type, QWidget *w, int parent, QString url)
    : TUIElement(id, type, w, parent, url)
{
	//connect()
	
}

/// Destructor
TUIWebview::~TUIWebview()
{
}

//void TUIWebview::valueChanged(bool)
//{
//    covise::TokenBuffer tb;
//    tb << ID;
//    TUIMainWindow::getInstance()->send(tb);
//}

void TUIWebview::urlChanged(QString url)
{
    covise::TokenBuffer tb;
    tb << ID;
    TUIMainWindow::getInstance()->send(tb);
}

const char *TUIWebview::getClassName() const
{
    return "TUIWebview";
}
