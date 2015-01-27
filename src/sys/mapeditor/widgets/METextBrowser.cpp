/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QDebug>

#include "METextBrowser.h"
#include "handler/MEMainHandler.h"

/*!
   \class METextBrowser
   \brief Modified textbrowser which can also handle external links (http://...)
*/

METextBrowser::METextBrowser(QWidget *parent)
    : QTextBrowser(parent)
    , m_baseUrl()
    , m_http()
{
    connect(&m_http, SIGNAL(requestFinished(int, bool)), this, SLOT(requestFinished(int, bool)));
}

//!
//! overwitten class can also handle external http requests
//!
QVariant METextBrowser::loadResource(int type, const QUrl &url)
{
    QByteArray data;

    // if url is relative (mostly for images) resolve it
    // otherwise store base url and fragment
    QUrl currentUrl;
    if (!url.isRelative())
    {
        fragment = url.fragment();
        m_baseUrl = url;
        currentUrl = url;
    }

    else
        currentUrl = m_baseUrl.resolved(url);

    // http request
    // start request & store stuff in a list
    if (currentUrl.scheme() == "http")
    {
        m_urlAndType uat;
        uat.m_type = type; // requested type
        uat.m_url = url; // requested url

        m_http.setHost(currentUrl.host());
        int httpGetId = m_http.get(currentUrl.path());
        m_requestedList.insert(httpGetId, uat);

        // show some text as hhtp request is asynchrounous
        return QVariant(QString("Please be patient, Try to load url ..."));
    }

    // local file request
    else
    {
        QFile f(currentUrl.toLocalFile());
        if (f.open(QFile::ReadOnly))
        {
            data = f.readAll();
            f.close();
        }

        else
        {
            qWarning("QTextBrowser: Cannot open '%s' for reading", url.toString().toLocal8Bit().data());
            return QVariant();
        }
    }

    return data;
}

//!
//! handle results from a http request
//!
void METextBrowser::requestFinished(int requestId, bool error)
{
    if (error)
        qCritical() << qPrintable(m_http.errorString());

    // show received content wgen id is in the list
    else
    {
        if (m_requestedList.contains(requestId))
        {
            // read data
            // get type and url for request
            QByteArray data = m_http.readAll();
            m_urlAndType uat = m_requestedList[requestId];

            // this is a image which has to be inserted into html text
            if (uat.m_type == QTextDocument::ImageResource)
            {
                QImage image;
                image.loadFromData(data);
                document()->addResource(QTextDocument::ImageResource, uat.m_url, image);
            }

            // normal http text
            else
            {
                setHtml(data);

                // scroll to a section html element
                if (!fragment.isEmpty())
                    scrollToAnchor(fragment);
            }
        }
    }
}

//!
//! document needs to be laid out again
//!
void METextBrowser::done(bool error)
{
    if (error)
        qCritical() << qPrintable(m_http.errorString());
}
