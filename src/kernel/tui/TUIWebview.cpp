/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIWebview.h"
#ifdef USE_WEBENGINE
#include "TUIMain.h"
#include <stdio.h>
#include <net/tokenbuffer.h>
#include <QQuickView>
#include <qmainwindow.h>

/// Constructor
TUIWebview::TUIWebview(int id, int type, QWidget *w, int parent, QString name)  //(QObject* parent, const std::string& n, int pID)
    : TUIElement(id, type, w, parent, name)
{
    Webview = new QWebEngineView(w);
    //Webview->load(QUrl("https://maps.google.com")); ///default website
    Webview->load(QUrl("http://vismobile.know-center.tugraz.at/#dashboardId=uLxmX&compressed=true&config=NobwRARgrgLjD2A7AkgEzALkjRBaADgJYCmAxgBYCGATjGADRgBuhAzgCqEwA2xmYABRIACAMJVawgBQBpSjGIBzeNRL1hAdWK0AlAzABbSviKJFmUGAqVEiYt34APXAEFHbfd0oR7-OQuVVPkZUeXkAT3w+LFYYVTN9SkVFaiV5QiR+MABfenBrW18scNd3Vk9vIrAtWn1QmAio-kQoAx9qROTUxXTMrEomc2yAXUYEfEwARkZeADM6DGmwAHdCVBhyTABmAA5GcmJCRXIFgFYAdkYAL2REVGJHfkn9VhUYACFwzHAvHwcsRKsUjEO6EBIYOJQYi5MBsd5HTCzSjcVjERhsACyYMIBkIV2I6AwSJRaLA9UoqJgADlKAZiOUMMBRiswah4Ms0PwWKwoMi8b08M9GNyAMoIVLfGGUUgwDKIfikVLyYgANQ8uXA0tlfUg1CgrE2jFRvBlBIAImELMMRkA"));
    WebviewLayout = new QHBoxLayout(w);
    WebviewLayout->addWidget(Webview);
    
    /* use instead of Webview->load()
    QWebEnginePage* page = Webview->page();
    page->load(QUrl(QStringLiteral("https://maps.google.com")));
    */
    connect(Webview, SIGNAL(loadFinished(bool)), this, SLOT(sendURL(bool)));
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
        const char* v;
        tb >> v; ///pointer to the begin of tranfered string (url) is saved in v 
        Webview->load(QUrl(v)); ///new url is loaded
    }
    else
    {
        TUIElement::setValue(type, tb);
    }
}

// send current url to cover
void TUIWebview::sendURL(bool)
{
    covise::TokenBuffer tb;
    tb << ID;
    int i = 1;
    tb << i;
    tb << Webview->url().path().toStdString().c_str();
    TUIMain::getInstance()->send(tb);
}
#endif
