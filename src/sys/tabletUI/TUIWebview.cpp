/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIWebview.h"
#include "TUIApplication.h"
#include <stdio.h>
#include <net/tokenbuffer.h>

/// Constructor
TUIWebview::TUIWebview(int id, int type, QWidget *w, int parent, QString name)  //(QObject* parent, const std::string& n, int pID)
    : TUIElement(id, type, w, parent, name)
{
    fprintf(stderr, "TUIWebview::TUIWebview\n");

    QWebEngineView* Webview = new QWebEngineView();
    Webview->load(QUrl("http://qt-project.org/"));
    Webview->show();
}

/// Destructor
TUIWebview::~TUIWebview()
{
}

const char *TUIWebview::getClassName() const
{
    return "TUIWebview";
}
