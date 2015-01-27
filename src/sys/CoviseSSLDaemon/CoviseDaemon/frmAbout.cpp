/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "frmAbout.h"

frmAbout::frmAbout(QApplication *app)
{
    mApplication = app;
}

frmAbout::~frmAbout()
{
}

void frmAbout::setupUi(QDialog *frmAbout)
{
    if (frmAbout->objectName().isEmpty())
        frmAbout->setObjectName(QString::fromUtf8("frmAbout"));
    QSize size(410, 182);
    size = size.expandedTo(frmAbout->minimumSizeHint());
    frmAbout->resize(size);
    label = new QLabel(frmAbout);
    label->setObjectName(QString::fromUtf8("label"));
    label->setGeometry(QRect(10, 10, 391, 131));
    QFont font;
    font.setFamily(QString::fromUtf8("DroidSansFallbackFull"));
    font.setPointSize(12);
    label->setFont(font);
    label->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
    btnClose = new QPushButton(frmAbout);
    btnClose->setObjectName(QString::fromUtf8("btnClose"));
    btnClose->setGeometry(QRect(130, 150, 131, 25));

    retranslateUi(frmAbout);

    QString qText = "Covise Daemon is part of the COVISE application suite.";
    qText += (char)13;
    qText += (char)13;
    qText += "It has been developed to replace the legacy RemoteDamon!";
    qText += (char)13;
    qText += "Other bla bla can go here!";
    this->label->setWordWrap(true);
    this->label->setText(qText);

    QMetaObject::connectSlotsByName(frmAbout);

    connectSlots();
} // setupUi

void frmAbout::retranslateUi(QDialog *frmAbout)
{
    frmAbout->setWindowTitle(QApplication::translate("frmAbout", "About", 0));
    label->setText(QApplication::translate("frmAbout", "Covise Daemon rev $Id$", 0));
    btnClose->setText(QApplication::translate("frmAbout", "Close", 0));
    Q_UNUSED(frmAbout);
} // retranslateUi

void frmAbout::connectSlots()
{
    connect(this->btnClose, SIGNAL(clicked(bool)), this, SLOT(handleClose(bool)));
}

void frmAbout::handleClose(bool)
{
    this->hide();
    delete this;
}
