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
class QWebEnginePage;
#elif defined(USE_WEBKIT)
class QWebView;
#elif defined(DUSE_TEXTBROWSER)
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

    void init();

private:
    void makeMenu();
#if defined(USE_WEBENGINE)
    QWebEngineView *m_browser = nullptr;
#elif defined(USE_WEBKIT)
    QWebView *m_browser = nullptr;
#elif defined(DUSE_TEXTBROWSER)
    METextBrowser *m_browser = nullptr;
#else
    QWidget *m_browser = nullptr;
#endif
    QAction *m_backwardId = nullptr, *m_forwardId = nullptr, *m_homeId = nullptr, *m_printId = nullptr;

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
