/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_TEXTBROWSER_H
#define ME_TEXTBROWSER_H

#include <QUrl>
#include <QTextBrowser>
//#include <QHttp>

//================================================
class METextBrowser : public QTextBrowser
//================================================
{
    Q_OBJECT

public:
    METextBrowser(QWidget *parent = 0);

private slots:

    void requestFinished(int requestID, bool error);
    void done(bool error);

private:
    struct m_urlAndType
    {
        int m_type;
        QUrl m_url;
    };

    QUrl m_baseUrl;
    //QHttp m_http;
    QString fragment;
    QMap<int, m_urlAndType> m_requestedList;

    QVariant loadResource(int type, const QUrl &url);
};
#endif
