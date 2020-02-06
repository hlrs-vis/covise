/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIWebview.h"
#ifdef USE_WEBENGINE
#include "TUIApplication.h"
#include <stdio.h>
#include <net/tokenbuffer.h>
#include <QQuickView>
#include <qmainwindow.h>

/// Constructor
TUIWebview::TUIWebview(int id, int type, QWidget *w, int parent, QString name)  //(QObject* parent, const std::string& n, int pID)
    : TUIElement(id, type, w, parent, name)
{
    fprintf(stderr, "TUIWebview::TUIWebview\n");

    Webview = new QWebEngineView(w);
    Webview->load(QUrl("https://maps.google.com")); ///default website
    WebviewLayout = new QHBoxLayout(w);
    WebviewLayout->addWidget(Webview);
    
    /* use instead of Webview->load()
    QWebEnginePage* page = Webview->page();
    page->load(QUrl(QStringLiteral("https://maps.google.com")));
    */

}

/// Destructor
TUIWebview::~TUIWebview()
{
}

const char *TUIWebview::getClassName() const
{
    return "TUIWebview";
}

void TUIWebview::setValue(TabletValue type, covise::TokenBuffer& tb) ///TUIWebview recieves a message from cover (tuiwebviewplugin)
{
    if (type == TABLET_STRING)
    {
        char* v;
        tb >> v; ///pointer to the begin of tranfered string (url) is saved in v 
        //cerr << "TUIWebview::setValue " << value << endl;
        Webview->load(QUrl(v)); ///new url is loaded
    }
    else
    {
        TUIElement::setValue(type, tb);
    }
}
#endif
