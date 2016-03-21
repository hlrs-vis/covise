/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_HELPWINDOW_H
#define ME_HELPWINDOW_H
#include <util/coTypes.h>

#include <QMainWindow>
class QComboBox;
class QUrl;
class QAction;

#ifndef QT_VERSION
#error "QT_VERSION not defined"
#endif

#if defined(USE_WEBENGINE)
class QWebEngineView;
#elif defined(USE_WEBKIT)
class QWebView;
#else
class METextBrowser;
#endif

//================================================
class MEHelpViewer : public QMainWindow
//================================================
{
    Q_OBJECT

public:
    MEHelpViewer();
    ~MEHelpViewer();

    static MEHelpViewer *instance();

    void init();

private:
    void makeMenu();
#if defined(USE_WEBENGINE)
    QWebEngineView *m_browser;
#elif defined(USE_WEBKIT)
    QWebView *m_browser;
#else
    METextBrowser *m_browser;
#endif
    QAction *m_backwardId, *m_forwardId, *m_homeId, *m_printId;

public slots:

    void newSource(const QString &);

private slots:

    void setBackwardAvailable(bool);
    void setForwardAvailable(bool);
    void linkClicked(const QUrl &);
    void print();
    void home();
};

#endif
