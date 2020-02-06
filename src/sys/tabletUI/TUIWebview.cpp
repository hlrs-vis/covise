/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIWebview.h"
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
    Webview->load(QUrl("https://maps.google.com"));
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

void TUIWebview::setValue(TabletValue type, covise::TokenBuffer& tb)
{
    if (type == TABLET_STRING)
    {
        char* v;
        tb >> v;
        //cerr << "TUIWebview::setValue " << value << endl;
        Webview->load(QUrl(v)); //Fehler QUrl anschauen, datentyp
    }
    else
    {
        TUIElement::setValue(type, tb);
    }
    
}


//methode load website (url) bekommt nachricht von webviewplugin
