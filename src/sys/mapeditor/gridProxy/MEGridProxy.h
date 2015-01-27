/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_GRIDPROXY_H
#define ME_GRIDPROXY_H

#include <QDialog>

#ifdef HAVE_GLOBUS
#undef IOV_MAX
#include "globus_common.h"
#include "globus_error.h"
#include "globus_gsi_cert_utils.h"
#include "globus_gsi_system_config.h"
#include "globus_gsi_proxy.h"
#include "globus_gsi_credential.h"
#include "globus_openssl.h"
extern "C" {
#include "myproxy.h"
}
#endif

class QLabel;
class QListBox;
class QLineEdit;
class QComboBox;
class QString;
class QGridLayout;
class QListBoxItem;
class QTextEdit;
class QProcess;
class QKeyEvent;

namespace covise
{
class coConfig;
}

class METable;

class coGlobusGridProxy
{

public:
    enum
    {
        PROXY_ERROR = 1,
        PROXY_COULDNT_INITIALIZE,
        PROXY_BAD_PASSPHRASE,
        PROXY_BAD_FILEIO,
        PROXY_COULDNT_CREATE_PROXY,
        PROXY_NOTMATCHING,
    };

    coGlobusGridProxy();
    ~coGlobusGridProxy();

    int createProxy(const char *user_cert_filename, const char *user_key_filename, const char *password);
    int getProxyCertificate(const char *user, const char *host, const char *password);
    int getProxyIdentity(char **subject, char **issuer, char **identity);
    time_t getProxyLifetime();
    int isFileCertificate(const char *file);

private:
#ifdef HAVE_GLOBUS
    globus_gsi_cred_handle_t proxy_cred_handle;
#endif
};

//================================================
class MEGridProxy : public QDialog
//================================================
{
    Q_OBJECT

public:
    MEGridProxy(QWidget *parent = 0);
    ~MEGridProxy();

private slots:

    int addCertRow();
    void createProxy();
    void deleteProxy();
    void queryMyProxy();
    void ok();
    void certFileBrowser();
    void getProxyInformation();

private:
    covise::coConfig *config;
    QLabel *status;
    QLabel *statusText;
    QTextEdit *proxy;
    QLineEdit *password;
    QLineEdit *myProxyUser;
    QLineEdit *myProxyHost;
    QLineEdit *myProxyPassword;
    METable *certTable;

    coGlobusGridProxy *gridProxy;
};
#endif
