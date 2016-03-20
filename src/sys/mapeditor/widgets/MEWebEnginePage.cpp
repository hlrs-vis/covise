#include "MEWebEnginePage.h"
#include <QDesktopServices>

MEWebEnginePage::MEWebEnginePage(QObject* parent)
: QWebEnginePage(parent)
{
}

bool MEWebEnginePage::acceptNavigationRequest(const QUrl & url, QWebEnginePage::NavigationType type, bool isMainFrame)
{
    if (type == QWebEnginePage::NavigationTypeLinkClicked)
    {
        if (url.host() != "fs.hlrs.de" || !url.path().startsWith("/projects/covise/doc"))
        {
            QDesktopServices::openUrl(url);
            return false;
        }
    }

    return true;
}
