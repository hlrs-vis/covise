/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QtGlobal>
#include <QTextEdit>
#include <QMessageBox>
#include <QDebug>
#include <QSslConfiguration>

#include <config/CoviseConfig.h>
#ifndef YAC
#include <net/covise_socket.h>
#else
#include <iostream>
#endif

#include "nodes/MECategory.h"
#include "hosts/MEHost.h"
#include "handler/MEMainHandler.h"

#include "MEApplication.h"
#ifndef Q_OS_MAC
// using MEApplication crashes on Ubuntu 10.4 with Qt 4.7.0
#define MEApplication QApplication
#endif

#if QT_VERSION < 0x040300
#error "Qt version 4.3.0 or higher is required"
#endif

#include <QMessageBox>

#if defined(Q_OS_MAC) || defined(Q_OS_LINUX)
#include <execinfo.h>
#endif

// With -Werror set, the original version from the qt library will
// cause compile time errors of the following kind:
// "error: format not a string literal and no format arguments
#define QT_REQUIRE_VERSION_PATCHED(argc, argv, str)                                                                                                                                                                                                                                                                                                     \
    {                                                                                                                                                                                                                                                                                                                                                   \
        QString s = QString::fromLatin1(str);                                                                                                                                                                                                                                                                                                           \
        QString sq = QString::fromLatin1(qVersion());                                                                                                                                                                                                                                                                                                   \
        if ((sq.section(QChar::fromLatin1('.'), 0, 0).toInt() << 16) + (sq.section(QChar::fromLatin1('.'), 1, 1).toInt() << 8) + sq.section(QChar::fromLatin1('.'), 2, 2).toInt() < (s.section(QChar::fromLatin1('.'), 0, 0).toInt() << 16) + (s.section(QChar::fromLatin1('.'), 1, 1).toInt() << 8) + s.section(QChar::fromLatin1('.'), 2, 2).toInt()) \
        {                                                                                                                                                                                                                                                                                                                                               \
            if (!qApp)                                                                                                                                                                                                                                                                                                                                  \
            {                                                                                                                                                                                                                                                                                                                                           \
                new MEApplication(argc, argv);                                                                                                                                                                                                                                                                                                          \
            }                                                                                                                                                                                                                                                                                                                                           \
            QString s = QApplication::tr("Executable '%1' requires Qt " "%2, found Qt %3.").arg(qAppName()).arg(QString::fromLatin1(str)).arg(QString::fromLatin1(qVersion()));                                                                                                                                                                         \
            QMessageBox::critical(0, QApplication::tr("Incompatible Qt Library Error"), s, QMessageBox::Abort, 0);                                                                                                                                                                                                                                      \
            qFatal("%s", s.toLatin1().data());                                                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                                               \
    }

//========================================================
// implement our own message handler
//========================================================
#if QT_VERSION >= 0x050000
void debugMsgHandler(QtMsgType type, const QMessageLogContext &, const QString &message)
#else
void debugMsgHandler(QtMsgType type, const char *message)
#endif
{
#if QT_VERSION >= 0x050000
    const QString &msg = message;
#else
    const QString msg(message);
#endif
#ifdef NDEBUG
    static bool useDebugWindow = covise::coCoviseConfig::isOn("System.MapEditor.DebugWindow", false);
#else
    static bool useDebugWindow = covise::coCoviseConfig::isOn("System.MapEditor.DebugWindow", false);
#endif
    static QTextEdit *edit = NULL;
    if (useDebugWindow && !edit)
    {
        edit = new QTextEdit();
        edit->show();
        edit->setWindowTitle(MEMainHandler::instance()->generateTitle("Debug Messages"));
    }

    switch (type)
    {
    case QtDebugMsg:
        std::cerr << "Debug: " << msg.toStdString() << std::endl;
        if (edit)
            edit->append(QString("<b>Debug:</b> %1").arg(msg));
        break;

    case QtWarningMsg:
        std::cerr << "Warning: " << msg.toStdString() << std::endl;
        if (edit)
            edit->append(QString("<b>Warning:</b> %1").arg(msg));
        break;

    case QtCriticalMsg:
        std::cerr << "Critical: " << msg.toStdString() << std::endl;
        if (edit)
            edit->append(QString("<font color=\"red\"> <b>Critical:</b> </font>%1").arg(msg));
        break;

    case QtFatalMsg:
        std::cerr << "Fatal: " << msg.toStdString() << std::endl;
        QMessageBox::critical(0, "Debug - Fatal", msg);
        break;
#if QT_VERSION >= 0x050500
    case QtInfoMsg:
        std::cerr << "Info: " << msg.toStdString() << std::endl;
        break;
#endif
    }

#if defined(Q_OS_MAC) || defined(Q_OS_LINUX)
    void *bt[500];
    int n = backtrace(bt, sizeof(bt) / sizeof(bt[0]));
    char **sym = backtrace_symbols(bt, n);
    std::cerr << "Location of error was:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        std::cerr << "   " << sym[i] << std::endl;
    }
    free(sym);
#endif
}

//========================================================
// main loop
//========================================================
int main(int argc, char **argv)
{
    QT_REQUIRE_VERSION_PATCHED(argc, argv, "4.3.0")
    std::cerr << "starting mapeditor" << std::endl;
    QSslSocket::addDefaultCaCertificates(QSslCertificate::fromPath(":/certs/telekom.pem"));
    QSslSocket::addDefaultCaCertificates(QSslCertificate::fromPath(":/certs/dfn.pem"));
    QSslSocket::addDefaultCaCertificates(QSslCertificate::fromPath(":/certs/uni-stuttgart.pem"));

    covise::Socket::initialize();

    // start user interface process
    MEApplication a(argc, argv);
    a.setWindowIcon(QIcon(":/icons/covise.png"));
    a.setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
    // this works around problems with messed layouts after settings fonts with qtconfig
    QApplication::setFont(QFont(QApplication::font().family(), QApplication::font().pointSize()));

    qRegisterMetaType<MECategory *>("MECategory");
    qRegisterMetaType<MEHost *>("MEHost");

#ifdef Q_OS_MAC
    a.setAttribute(Qt::AA_DontShowIconsInMenus);
#endif
#if QT_VERSION >= 0x050000
    a.setAttribute(Qt::AA_UseHighDpiPixmaps, true);
#endif
#if QT_VERSION >= 0x050600
    a.setAttribute(Qt::AA_EnableHighDpiScaling, true);
#endif

    //DebugBreak();
    new MEMainHandler(argc, argv);

// this has to be done after creating MEMainHandler - generateTitle depends on it
#if QT_VERSION >= 0x050000
    qInstallMessageHandler(debugMsgHandler);
#else
    qInstallMsgHandler(debugMsgHandler);
#endif

    a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));
    return a.exec();
}
