/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "frmRequestDialog.h"

frmRequestDialog::frmRequestDialog()
{
}

frmRequestDialog::~frmRequestDialog()
{
}

void frmRequestDialog::setupUi(QDialog *frmRequestConfirm)
{
    if (frmRequestConfirm->objectName().isEmpty())
        frmRequestConfirm->setObjectName(QString::fromUtf8("frmRequestConfirm"));
    QSize size(409, 105);
    size = size.expandedTo(frmRequestConfirm->minimumSizeHint());
    frmRequestConfirm->resize(size);
    grpBorder = new QGroupBox(frmRequestConfirm);
    grpBorder->setObjectName(QString::fromUtf8("grpBorder"));
    grpBorder->setGeometry(QRect(10, 10, 391, 51));
    lblMessage = new QLabel(grpBorder);
    lblMessage->setObjectName(QString::fromUtf8("lblMessage"));
    lblMessage->setGeometry(QRect(10, 10, 381, 41));
    lblMessage->setAlignment(Qt::AlignLeading | Qt::AlignLeft | Qt::AlignTop);
    lblMessage->setWordWrap(true);
    layoutWidget = new QWidget(frmRequestConfirm);
    layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
    layoutWidget->setGeometry(QRect(60, 50, 260, 47));
    hboxLayout = new QHBoxLayout(layoutWidget);
    hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
    hboxLayout->setContentsMargins(0, 0, 0, 0);
    btnAllow = new QPushButton(layoutWidget);
    btnAllow->setObjectName(QString::fromUtf8("btnAllow"));

    hboxLayout->addWidget(btnAllow);

    btnTempAllow = new QPushButton(layoutWidget);
    btnTempAllow->setObjectName(QString::fromUtf8("btnTempAllow"));

    hboxLayout->addWidget(btnTempAllow);

    btnDeny = new QPushButton(layoutWidget);
    btnDeny->setObjectName(QString::fromUtf8("btnDeny"));

    hboxLayout->addWidget(btnDeny);

    retranslateUi(frmRequestConfirm);

    QMetaObject::connectSlotsByName(frmRequestConfirm);
    connect(this->btnTempAllow, SIGNAL(clicked(bool)), this, SLOT(handleTemporary(bool)));
    connect(this->btnAllow, SIGNAL(clicked(bool)), this, SLOT(handlePermanent(bool)));
    connect(this->btnDeny, SIGNAL(clicked(bool)), this, SLOT(handleDeny(bool)));
} // setupUi

void frmRequestDialog::retranslateUi(QDialog *frmRequestConfirm)
{
    this->setWindowModality(Qt::WindowModal);
    frmRequestConfirm->setWindowTitle(QApplication::translate("frmRequestConfirm", "Remote application launch request", 0));
    grpBorder->setTitle(QString());
    lblMessage->setText(QApplication::translate("frmRequestConfirm", "Someone from PX3459 tries to start COVISE on your machine ..... Allow this?", 0));
    btnAllow->setToolTip(QApplication::translate("frmRequestConfirm", "This adds the remote host and partner to your permanent list of allowed hosts and partners.", 0));
    btnAllow->setText(QApplication::translate("frmRequestConfirm", "Allow", 0));
    btnTempAllow->setToolTip(QApplication::translate("frmRequestConfirm", "This allows the application launch for this request without storing any information about the remote site", 0));
    btnTempAllow->setText(QApplication::translate("frmRequestConfirm", "Allow temporarily", 0));
    btnDeny->setToolTip(QApplication::translate("frmRequestConfirm", "Deny the request totally, not launching anything", 0));
    btnDeny->setText(QApplication::translate("frmRequestConfirm", "Deny", 0));
    Q_UNUSED(frmRequestConfirm);
} // retranslateUi

void frmRequestDialog::setMessage(const char *message, DialogMode mode)
{
    lblMessage->setText(QApplication::translate("frmRequestConfirm", message, 0));
    lblMessage->update();
    mCurrentMode = mode;
}

frmRequestDialog::DialogMode frmRequestDialog::getCurrentMode()
{
    return mCurrentMode;
}

void frmRequestDialog::handlePermanent(bool)
{
    if (mSSLObject)
    {
        mSSLObject->allowPermanent(false);
    }
    this->hide();
}

void frmRequestDialog::handleTemporary(bool)
{
    if (mSSLObject)
    {
        mSSLObject->allow();
    }
    this->hide();
}

void frmRequestDialog::handleDeny(bool)
{
    if (mSSLObject)
    {
        mSSLObject->deny();
    }
    this->hide();
}

void frmRequestDialog::setHandleObject(SSLDaemon *object)
{
    mSSLObject = object;
}

void frmRequestDialog::setDaemon(SSLDaemon *ssl)
{
    this->mSSLObject = ssl;
}
