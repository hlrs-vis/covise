/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QLabel>
#include <QTimer>
#include <QPushButton>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QTime>
#include <QFileDialog>

#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

#include "MEGridProxy.h"
#include "METable.h"
#include "widgets/MEUserInterface.h"
#include "handler/MEMainHandler.h"

#ifdef HAVE_GLOBUS
#include "globus_common.h"
#include "globus_error.h"
#include "globus_gsi_cert_utils.h"
#include "globus_gsi_system_config.h"
#include "globus_gsi_proxy.h"
#include "globus_gsi_credential.h"
#include "proxycertinfo.h"
#include "openssl/asn1.h"
static char globus_password[80];
#endif

using namespace std;

/*****************************************************************************
 *
 * Class MEGridProxy
 *
 *****************************************************************************/

MEGridProxy::MEGridProxy(QWidget *parent)
    : QDialog(parent)
{

    setWindowTitle("Globus Grid Proxy");

#ifdef HAVE_GLOBUS
    gridProxy = new coGlobusGridProxy();
#else
    gridProxy = NULL;
#endif

    config = covise::coConfig::getInstance();

    QVBoxLayout *fbox = new QVBoxLayout(this);

    QLabel *l_proxy = new QLabel(this);
    l_proxy->setText("Current Grid Proxy Identity:");
    fbox->addWidget(l_proxy);

    fbox->addSpacing(5);

    proxy = new QTextEdit(this);
    proxy->setReadOnly(true);
    getProxyInformation();
    fbox->addWidget(proxy);

    QHBoxLayout *hbox = new QHBoxLayout(this);
    fbox->addLayout(hbox);
    QPushButton *deleteProxy = new QPushButton("&Delete Proxy", this);
    connect(deleteProxy, SIGNAL(clicked()), this, SLOT(deleteProxy()));
    hbox->addStretch(1);
    hbox->addWidget(deleteProxy);

    fbox->addSpacing(5);

    QTabWidget *tabs = new QTabWidget(this);

    QWidget *localTab = new QWidget(tabs);
    tabs->insertTab(0, localTab, "Local Certificate");

    QWidget *myProxyTab = new QWidget(tabs);
    tabs->insertTab(1, myProxyTab, "Globus MyProxy");
    resize(QSize(600, 480).expandedTo(minimumSizeHint()));

    // content localtab
    QVBoxLayout *lfbox = new QVBoxLayout(localTab);
    hbox = new QHBoxLayout(localTab);
    lfbox->addLayout(hbox);
    QLabel *l_list = new QLabel(localTab);
    l_list->setText("List of Certificates:");
    hbox->addWidget(l_list);

    QPushButton *certBrowser = new QPushButton(localTab);
    certBrowser->setIcon(QPixmap(":/icons/folder_downloads.png"));
    connect(certBrowser, SIGNAL(clicked()), this, SLOT(certFileBrowser()));
    hbox->addStretch(10);
    hbox->addWidget(certBrowser);
    certBrowser->setToolTip("Load Certificate or Key");
    QPushButton *addCert = new QPushButton(localTab);
    addCert->setIcon(QPixmap(":/icons/master.xpm"));
    addCert->setToolTip("Insert new Certificate/Key Pair");
    connect(addCert, SIGNAL(clicked()), this, SLOT(addCertRow()));
    hbox->addWidget(addCert);

    certTable = new METable(0, 2, localTab);
    certTable->setShowGrid(false);
    QStringList hheader;
    hheader << "Certificate"
            << "Key";
    certTable->setHorizontalHeaderLabels(hheader);
    certTable->setColumnWidth(0, 280);
    certTable->setColumnWidth(1, 280);
    lfbox->addWidget(certTable);

    char *cert = 0, *key = 0;

#ifdef HAVE_GLOBUS
    globus_result_t result = GLOBUS_GSI_SYSCONFIG_GET_USER_CERT_FILENAME(&cert, &key);
    if (result != GLOBUS_SUCCESS)
        fprintf(stderr, "ERROR: Couldn't find valid credentials to generate a proxy.\n");
    else
    {
        certTable->setItem(0, 0, new QTableWidgetItem(cert));
        certTable->setItem(0, 1, new QTableWidgetItem(key));
    }
#endif

    int i = 0;
    covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries("GLOBUS.certificates");
    const char **line = entries.getValue();
    if (line)
    {
        while (line[i] != NULL)
        {
            int add = 1;
            char buf[128];
            snprintf(buf, 128, "GLOBUS.certificates.%s", line[i]);
            std::string confCert = covise::coCoviseConfig::getEntry("cert", buf);
            std::string confKey = covise::coCoviseConfig::getEntry("key", buf);

            if (cert && key && !confCert.empty() && !confKey.empty())
                if (!strcmp(cert, confCert.c_str()) && !strcmp(key, confKey.c_str()))
                    add = 0;

            if (add)
            {
                int num = addCertRow();
                certTable->setItem(num - 1, 0, new QTableWidgetItem(confCert.c_str()));
                certTable->setItem(num - 1, 1, new QTableWidgetItem(confKey.c_str()));
            }

            i += 2;
        }
    }
    else
    {
        MEUserInterface::instance()->printMessage("no certificates");
    }

    if (cert)
        free(cert);
    if (key)
        free(key);

    hbox = new QHBoxLayout(localTab);
    lfbox->addLayout(hbox);

    QLabel *l_password = new QLabel(localTab);
    l_password->setText("Password for private key:");
    hbox->addWidget(l_password);

    password = new QLineEdit(localTab);
    password->setEchoMode(QLineEdit::Password);
    hbox->addWidget(password);

    QPushButton *newProxy = new QPushButton("&Create Proxy", localTab);
    hbox->addWidget(newProxy);
    connect(newProxy, SIGNAL(clicked()), this, SLOT(createProxy()));

    // content myProxyTab
    QVBoxLayout *mfbox = new QVBoxLayout(myProxyTab);

    hbox = new QHBoxLayout(myProxyTab);
    mfbox->addLayout(hbox);

    QLabel *myProxyLabel = new QLabel(myProxyTab);
    myProxyLabel->setText("Retrieve a proxy certificate from a Globus MyProxy Service");
    hbox->addWidget(myProxyLabel);
    mfbox->addSpacing(5);

    hbox = new QHBoxLayout(myProxyTab);
    mfbox->addLayout(hbox);

    QLabel *myProxyUserLabel = new QLabel(myProxyTab);
    QLabel *myProxyHostLabel = new QLabel(myProxyTab);

    myProxyUserLabel->setText("MyProxy User:");
    hbox->addWidget(myProxyUserLabel);

    myProxyUser = new QLineEdit(myProxyTab);
    hbox->addWidget(myProxyUser);

    hbox = new QHBoxLayout(myProxyTab);
    mfbox->addLayout(hbox);

    myProxyHostLabel->setText("MyProxy Host:");
    hbox->addWidget(myProxyHostLabel);

    myProxyHost = new QLineEdit(myProxyTab);
    hbox->addWidget(myProxyHost);

    std::string myProxyUserConfig = covise::coCoviseConfig::getEntry("user", "GLOBUS.MyProxy");
    std::string myProxyHostConfig = covise::coCoviseConfig::getEntry("host", "GLOBUS.MyProxy");
    myProxyUser->setText(myProxyUserConfig.c_str());
    myProxyHost->setText(myProxyHostConfig.c_str());

    hbox = new QHBoxLayout(myProxyTab);
    mfbox->addLayout(hbox);

    QLabel *myProxyPasswordLabel = new QLabel(myProxyTab);
    myProxyPasswordLabel->setText("MyProxy Certificate Password:");
    hbox->addWidget(myProxyPasswordLabel);

    myProxyPassword = new QLineEdit(myProxyTab);
    myProxyPassword->setEchoMode(QLineEdit::Password);
    hbox->addWidget(myProxyPassword);

    QPushButton *queryMyProxy = new QPushButton("&Query Myproxy", myProxyTab);
    hbox->addWidget(queryMyProxy);
    connect(queryMyProxy, SIGNAL(clicked()), this, SLOT(queryMyProxy()));
    mfbox->addStretch(1);

    fbox->addWidget(tabs);

    hbox = new QHBoxLayout(this);
    fbox->addLayout(hbox);

    status = new QLabel(this);
    //status->setAlignment(Qt::AlignHCenter);
    hbox->addWidget(status);

    hbox = new QHBoxLayout(this);
    fbox->addLayout(hbox);

    QPushButton *ok = new QPushButton("&Ok", this);
    hbox->addStretch(1);
    hbox->addWidget(ok);
    connect(ok, SIGNAL(clicked()), this, SLOT(ok()));

    QTimer *timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(getProxyInformation()));
    timer->start(10000);

    setWindowIcon(MEMainHandler::instance()->pm_logo);
}

MEGridProxy::~MEGridProxy()
{

#ifdef HAVE_GLOBUS
    delete gridProxy;
    gridProxy = NULL;
#endif
}

void MEGridProxy::getProxyInformation()
{

    char *subject = NULL;
    char *issuer = NULL;
    char *identity = NULL;
    time_t lifetime;

    proxy->clear();
    if (gridProxy)
    {
        int result = gridProxy->getProxyIdentity(&subject, &issuer, &identity);

        if (result)
        {
            lifetime = gridProxy->getProxyLifetime();
            if (lifetime <= 0)
            {
                lifetime = 0;
                proxy->setFontWeight(QFont::Bold);
                proxy->setTextColor(QColor(255, 0, 0));
                proxy->append("Proxy Lifetime: expired");
            }
            else
            {
                QTime time = QTime(lifetime / 3600, (lifetime % 3600) / 60, lifetime % 60);
                proxy->setFontWeight(QFont::Bold);
                proxy->append("Proxy Lifetime: " + time.toString());
            }
            proxy->setFontWeight(QFont::Normal);
            proxy->setTextColor(QColor(0, 0, 0));
            proxy->append(QString("Subject: ") + QString(subject));
            proxy->append(QString("Issuer: ") + QString(issuer));
            proxy->append(QString("Identity: ") + QString(identity));

            delete[] subject;
            delete[] issuer;
            delete[] identity;
        }
        else
        {
            proxy->setTextColor(QColor(255, 0, 0));
            proxy->setFontWeight(QFont::Bold);
            proxy->append("No valid grid proxy found.");
            proxy->setFontWeight(QFont::Normal);
            proxy->setTextColor(QColor(0, 0, 0));
        }
    }
    else
    {
        proxy->setTextColor(QColor(255, 0, 0));
        proxy->setFontWeight(QFont::Bold);
        proxy->append("COVISE has been compiled without GLOBUS support.");
        proxy->setFontWeight(QFont::Normal);
        proxy->setTextColor(QColor(0, 0, 0));
    }
}

void MEGridProxy::certFileBrowser()
{

    QString file = QFileDialog::getOpenFileName(this, "Choose a file to open", QDir::homePath(), "GLOBUS Certificates (*.pem)");

    if (file.isEmpty())
        return;

    QList<QTableWidgetItem *> list = certTable->selectedItems();
    int n = 0;
    if (!list.isEmpty())
    {
        n = certTable->row(list[0]);
    }
    /*while (n < certTable->rowCount()) {
      if (certTable->isRowSelected(n))
         break;
      n++;
   }*/

    if (n == certTable->rowCount())
        return;

    int column = 0;
    if (gridProxy)
    {
        QByteArray ba = file.toLatin1();
        if (!gridProxy->isFileCertificate(ba.data()))
            column = 1;
    }

    certTable->setItem(n, column, new QTableWidgetItem(file));
}

int MEGridProxy::addCertRow()
{

    certTable->insertRow(certTable->rowCount());
    certTable->setItem(certTable->rowCount() - 1, 0, new QTableWidgetItem());
    certTable->setItem(certTable->rowCount() - 1, 1, new QTableWidgetItem());
    return certTable->rowCount();
}

void MEGridProxy::deleteProxy()
{
#ifdef HAVE_GLOBUS
    globus_result_t result = GLOBUS_SUCCESS;
    char *proxy_filename = 0;

    result = GLOBUS_GSI_SYSCONFIG_GET_PROXY_FILENAME(&proxy_filename,
                                                     GLOBUS_PROXY_FILE_INPUT);

    if (result == GLOBUS_SUCCESS)
    {
        unlink(proxy_filename);
        status->setText("Grid Proxy deleted.");
        getProxyInformation();
    }

    delete[] proxy_filename;
#endif
}

void MEGridProxy::queryMyProxy()
{

#ifdef HAVE_GLOBUS
    if (gridProxy)
    {
        QString user = myProxyUser->text();
        QString host = myProxyHost->text();
        QString password = myProxyPassword->text();

        int result = gridProxy->getProxyCertificate(user.toLatin1().data(), host.toLatin1().data(), password.toLatin1().data());

        switch (result)
        {

        case 0:
            status->setText("Grid Proxy created.");
            getProxyInformation();
            break;
        case 1:
            status->setText("Could not connect to service.");
            break;
        case 2:
            status->setText("Could not receive credentials.");
            break;

        default:
            break;
        }
        myProxyPassword->clear();
    }
#endif
}

void MEGridProxy::createProxy()
{

    if (certTable->rowCount() < 1)
    {
        status->setText("No certificate/key pair selected.");
        return;
    }

    int n = 0;
    QList<QTableWidgetItem *> list = certTable->selectedItems();
    if (!list.isEmpty())
    {
        n = certTable->row(list[0]);
    }
    /*while (n < certTable->rowCount()) {
      if (certTable->isRowSelected(n))
         break;
      n++;
   }*/

    if (n == certTable->rowCount())
        n = 0;

#ifdef HAVE_GLOBUS
    if (gridProxy)
    {
        QString cert = certTable->item(n, 0)->text();
        QString key = certTable->item(n, 1)->text();

        //if (!cert.isNull() && !key.isNull())
        {
            switch (gridProxy->createProxy(cert.toLatin1().data(), key.toLatin1().data(), password->text().toLatin1().data()))
            {
            case 0:
                status->setText("Grid Proxy created.");
                break;
            case coGlobusGridProxy::PROXY_COULDNT_INITIALIZE:
                status->setText("Could not initialize Grid Proxy.");
                break;
            case coGlobusGridProxy::PROXY_BAD_PASSPHRASE:
                status->setText("Could not initialize Grid Proxy: Bad Passphrase supplied.");
                break;
            case coGlobusGridProxy::PROXY_BAD_FILEIO:
                status->setText("Could not initialize Grid Proxy: Check file permissions.");
                break;
            case coGlobusGridProxy::PROXY_COULDNT_CREATE_PROXY:
                status->setText("Could not initialize Grid Proxy.");
                break;
            case coGlobusGridProxy::PROXY_NOTMATCHING:
                status->setText("Could not initialize Grid Proxy: Certificate and Key do not match.");
                break;
            case coGlobusGridProxy::PROXY_ERROR:
            default:
                status->setText("GLOBUS Grid Proxy could not be created.");
            }

            password->clear();
        }
    }
#else
    status->setText("COVISE was compiled without GLOBUS support.");
#endif

    getProxyInformation();
}

void MEGridProxy::ok()
{
    hide();
}

#ifdef HAVE_GLOBUS
static int globus_password_callback(char *buf, int num, int w)
{

    w = 0;
    int i = strlen(globus_password);
    strncpy(buf, globus_password, num);
    buf[i] = '\0';
    return i;
}
#endif

coGlobusGridProxy::coGlobusGridProxy()
{

#ifdef HAVE_GLOBUS
    if (globus_module_activate(GLOBUS_GSI_PROXY_MODULE) != (int)GLOBUS_SUCCESS)
        fprintf(stderr, "ERROR: Couldn't load module: GLOBUS_GSI_PROXY_MODULE.\nMake sure Globus is installed correctly.\n");
#endif
}

coGlobusGridProxy::~coGlobusGridProxy()
{
#ifdef HAVE_GLOBUS
    globus_gsi_cred_handle_destroy(proxy_cred_handle);
    globus_module_deactivate_all();
#endif
}

int coGlobusGridProxy::getProxyIdentity(char **subject, char **issuer, char **identity)
{
#ifdef HAVE_GLOBUS
    globus_result_t result = GLOBUS_SUCCESS;
    char *proxy_filename;

    result = GLOBUS_GSI_SYSCONFIG_GET_PROXY_FILENAME(&proxy_filename,
                                                     GLOBUS_PROXY_FILE_INPUT);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "coGlobusGridProxy: couldn't find a valid proxy\n");
        return 0;
    }

    result = globus_gsi_cred_handle_init(&proxy_cred_handle, NULL);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "coGlobusGridProxy: couldn't initialize proxy credential handle\n");
        return 0;
    }

    result = globus_gsi_cred_read_proxy(proxy_cred_handle, proxy_filename);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "coGlobusGridProxy: couldn't read proxy from: %s\n", proxy_filename);
        return 0;
    }

    result = globus_gsi_cred_get_subject_name(proxy_cred_handle, subject);

    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "coGlobusGridProxy: couldn't get valid subject from the proxy credential\n");
        return 0;
    }

    result = globus_gsi_cred_get_issuer_name(proxy_cred_handle, issuer);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "coGlobusGridProxy: couldn't get valid issuer from the proxy credential\n");
        return 0;
    }

    result = globus_gsi_cred_get_identity_name(proxy_cred_handle, identity);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "coGlobusGridProxy: couldn't get valid  identity from the proxy credential\n");
        return 0;
    }

    return 1;
#else
    *subject = *issuer = *identity = 0;
    return 0;
#endif
}

time_t coGlobusGridProxy::getProxyLifetime()
{
#ifdef HAVE_GLOBUS
    globus_result_t result = GLOBUS_SUCCESS;
    time_t lifetime;

    result = globus_gsi_cred_get_lifetime(proxy_cred_handle,
                                          &lifetime);
    if (result != GLOBUS_SUCCESS)
        fprintf(stderr, "ERROR: Can't get the lifetime of the proxy credential.\n");

    return lifetime;
#else
    return 0;
#endif
}

int coGlobusGridProxy::isFileCertificate(const char *file)
{
#ifdef HAVE_GLOBUS
    globus_result_t result = GLOBUS_SUCCESS;
    globus_gsi_cred_handle_attrs_t cred_handle_attrs = NULL;
    globus_gsi_cred_handle_t cred_handle = NULL;

    result = globus_gsi_cred_handle_attrs_init(&cred_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't initialize credential handle attributes\n");
        return 0;
    }

    result = globus_gsi_cred_handle_init(&cred_handle, cred_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't initialize credential handle\n");
        return 0;
    }

    result = globus_gsi_cred_handle_attrs_destroy(cred_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        globus_libc_fprintf(stderr, "ERROR: Couldn't destroy credential handle attributes.\n");
        return 0;
    }

    result = globus_gsi_cred_read_cert(cred_handle, (char *)file);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't read user certificate\ncert file location: %s\n\n", file);
        return 0;
    }

    return 1;
#else
    // don't generate unused parameter warning
    (void)file;
    return 0;
#endif
}

int coGlobusGridProxy::createProxy(const char *user_cert_filename, const char *user_key_filename, const char *password)
{
#ifdef HAVE_GLOBUS
    char *proxy_out_filename = NULL;
    globus_gsi_cred_handle_attrs_t cred_handle_attrs = NULL;
    globus_gsi_cred_handle_t cred_handle = NULL;

    globus_gsi_proxy_handle_t proxy_handle = NULL;
    globus_gsi_proxy_handle_attrs_t proxy_handle_attrs = NULL;
    int key_bits = 512;
    int valid = 12 * 60;
    globus_gsi_cert_utils_cert_type_t cert_type = GLOBUS_GSI_CERT_UTILS_TYPE_GSI_3_IMPERSONATION_PROXY;
    globus_result_t result = GLOBUS_SUCCESS;

    strncpy(globus_password, password, 79);

    result = globus_gsi_proxy_handle_attrs_init(&proxy_handle_attrs);

    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't initialize the proxy handle attributes.\n");
        return PROXY_COULDNT_INITIALIZE;
    }

    result = globus_gsi_proxy_handle_attrs_set_keybits(proxy_handle_attrs,
                                                       key_bits);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't set the key bits for the private key of the proxy certificate\n");
        return PROXY_ERROR;
    }

    result = globus_gsi_proxy_handle_init(&proxy_handle, proxy_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't initialize the proxy handle\n");
        return PROXY_COULDNT_INITIALIZE;
    }

    result = globus_gsi_proxy_handle_attrs_destroy(proxy_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't destroy proxy handle attributes.\n");
        return PROXY_ERROR;
    }

    result = globus_gsi_proxy_handle_set_time_valid(proxy_handle, valid);

    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't set the validity time of the proxy cert to %d minutes.\n", valid);
        return PROXY_ERROR;
    }

    result = globus_gsi_proxy_handle_set_type(proxy_handle, cert_type);

    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't set the type of the proxy cert\n");
        return PROXY_ERROR;
    }

    result = GLOBUS_GSI_SYSCONFIG_GET_PROXY_FILENAME(&proxy_out_filename,
                                                     GLOBUS_PROXY_FILE_OUTPUT);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't find a valid location to write the proxy file\n");
        return PROXY_BAD_FILEIO;
    }
    //printf("proxy_file: [%s]\n", proxy_out_filename);

    result = globus_gsi_cred_handle_attrs_init(&cred_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't initialize credential handle attributes\n");
        return PROXY_ERROR;
    }

    result = globus_gsi_cred_handle_init(&cred_handle, cred_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't initialize credential handle\n");
        return PROXY_ERROR;
    }

    result = globus_gsi_cred_handle_attrs_destroy(cred_handle_attrs);
    if (result != GLOBUS_SUCCESS)
    {
        globus_libc_fprintf(stderr, "ERROR: Couldn't destroy credential handle attributes.\n");
        return PROXY_ERROR;
    }

    result = globus_gsi_cred_read_cert(cred_handle, (char *)user_cert_filename);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Couldn't read user certificate cert file location: [%s]\n", user_cert_filename);
        return PROXY_BAD_FILEIO;
    }

    result = globus_gsi_cred_read_key(cred_handle,
                                      (char *)user_key_filename,
                                      (int (*)())globus_password_callback);
    if (result != GLOBUS_SUCCESS)
    {

        globus_object_t *error;
        error = globus_error_get(result);
        fprintf(stderr, "ERROR: Couldn't read user key: Bad passphrase. key file location: %s\n", user_key_filename);
        return PROXY_BAD_PASSPHRASE;

        /*
      if (globus_error_match_openssl_error(error,
                                           ERR_LIB_PEM,
                                           PEM_F_PEM_DO_HEADER,
                                           PEM_R_BAD_DECRYPT) == GLOBUS_TRUE) {
         fprintf(stderr, "ERROR: Couldn't read user key: Bad passphrase. key file location: %s\n", user_key_filename);
         return PROXY_BAD_PASSPHRASE;
      } else {
         fprintf(stderr, "ERROR: Couldn't read user key. key file location: %s\n", user_key_filename);
         return PROXY_BAD_FILEIO;
      }
      */
    }

    result = globus_gsi_proxy_create_signed(proxy_handle, cred_handle, &proxy_cred_handle);
    if (result != GLOBUS_SUCCESS)
        fprintf(stderr, "ERROR: Couldn't create proxy certificate\n");

    result = globus_gsi_cred_verify(proxy_cred_handle);

    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: Could not verify the signature of the generated proxy certificate\n       This is likely due to a non-matching user key and cert\n\n");
        return PROXY_NOTMATCHING;
    }
    result = globus_gsi_cred_write_proxy(proxy_cred_handle,
                                         proxy_out_filename);
    if (result != GLOBUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: The proxy credential could not be written to the output file.\n");
        return PROXY_BAD_FILEIO;
    }
    return 0;
#else
    // don't generate unused parameter warning

    (void)user_cert_filename;
    (void)user_key_filename;
    (void)password;
    return PROXY_ERROR;
#endif
}

int coGlobusGridProxy::getProxyCertificate(const char *user, const char *host, const char *password)
{

#ifdef HAVE_GLOBUS
    char *outputfile = NULL;
    myproxy_socket_attrs_t *socket_attrs;
    myproxy_request_t *client_request;
    myproxy_response_t *server_response;

    socket_attrs = (myproxy_socket_attrs_t *)malloc(sizeof(*socket_attrs));
    memset(socket_attrs, 0, sizeof(*socket_attrs));

    client_request = (myproxy_request_t *)malloc(sizeof(*client_request));
    memset(client_request, 0, sizeof(*client_request));

    server_response = (myproxy_response_t *)malloc(sizeof(*server_response));
    memset(server_response, 0, sizeof(*server_response));

    myproxy_set_delegation_defaults(socket_attrs, client_request);

    //client_request->proxy_lifetime = 0;
    client_request->username = strdup(user);
    socket_attrs->pshost = strdup(host);
    //socket_attrs->psport = 0;

    /* Connect to server. */
    if (myproxy_init_client(socket_attrs) < 0)
    {
        fprintf(stderr, "Error: %s\n", verror_get_string());
        myproxy_free(socket_attrs, client_request, server_response);
        return 1;
    }

    globus_module_activate(GLOBUS_GSI_SYSCONFIG_MODULE);
    GLOBUS_GSI_SYSCONFIG_GET_PROXY_FILENAME(&outputfile,
                                            GLOBUS_PROXY_FILE_OUTPUT);

    int length = strlen(password);
    strncpy(client_request->passphrase, password, length <= 1024 ? length : 1024);

    if (myproxy_get_delegation(socket_attrs, client_request, NULL,
                               server_response, outputfile))
    {
        fprintf(stderr, "Failed to receive credentials.\n");
        verror_print_error(stderr);
        myproxy_free(socket_attrs, client_request, server_response);
        return 2;
    }

    myproxy_free(socket_attrs, client_request, server_response);
#else
    (void)host;
    (void)user;
    (void)password;
#endif
    return 0;
}
