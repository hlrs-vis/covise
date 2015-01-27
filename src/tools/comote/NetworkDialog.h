/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// NetworkDialog.h

#ifndef NETWORK_DIALOG_H
#define NETWORK_DIALOG_H

#include <QDialog>

#include "ui_NetworkDialog.h"

class NetworkDialog : public QDialog
{
    Q_OBJECT

public:
    NetworkDialog(QWidget *parent = 0);

    virtual ~NetworkDialog();

    // Returns the hostname
    QString getHostname();

    // Returns the port
    unsigned short getTCPPort();
    unsigned short getUDPPort();

private slots:
    void onAccepted();

private:
    void addToComboBox(QComboBox *box, const QString &text);

    void parseHostname(FILE *file, QComboBox *box);
    void parsePort(FILE *file, QComboBox *box);

    void writeComboBox(FILE *file, const char *prefix, const QComboBox *box);

    void readConfig();
    void writeConfig();

private:
    Ui::NetworkDialogUi ui;
};
#endif
