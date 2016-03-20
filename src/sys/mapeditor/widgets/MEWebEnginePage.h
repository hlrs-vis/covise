#ifndef MEWEBENGINEPAGE_H
#define MEWEBENGINEPAGE_H

#include <QWebEnginePage>

class MEWebEnginePage : public QWebEnginePage
{
    Q_OBJECT
public:
    MEWebEnginePage(QObject* parent = 0);
    bool acceptNavigationRequest(const QUrl & url, QWebEnginePage::NavigationType type, bool isMainFrame);
};

#endif
