/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// NetworkDialog.cpp

#include "NetworkDialog.h"
#include "Debug.h"

#include <stdio.h>
#include <string.h>

NetworkDialog::NetworkDialog(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);

#if !USE_TUIO
    ui.labelUDPPort->setVisible(false);
    ui.comboBoxUDPPort->setVisible(false);
#endif

    // The contents of the combo boxes is only updated if the combo box
    // has the keyboard focus. Update manually.
    connect(this, SIGNAL(accepted()), this, SLOT(onAccepted()));

    readConfig();
}

NetworkDialog::~NetworkDialog()
{
    writeConfig();
}

QString NetworkDialog::getHostname()
{
    return ui.comboBoxHostname->currentText();
}

unsigned short NetworkDialog::getTCPPort()
{
    return ui.comboBoxTCPPort->currentText().toInt();
}

unsigned short NetworkDialog::getUDPPort()
{
#if USE_TUIO
    return ui.comboBoxUDPPort->currentText().toInt();
#else
    return 0;
#endif
}

void NetworkDialog::onAccepted()
{
    addToComboBox(ui.comboBoxHostname, ui.comboBoxHostname->currentText());
    addToComboBox(ui.comboBoxTCPPort, ui.comboBoxTCPPort->currentText());
#if USE_TUIO
    addToComboBox(ui.comboBoxUDPPort, ui.comboBoxUDPPort->currentText());
#endif
}

void NetworkDialog::addToComboBox(QComboBox *box, const QString &text)
{
    if (box->findText(text) == -1)
    {
        box->addItem(text);
    }
}

void NetworkDialog::parseHostname(FILE *file, QComboBox *box)
{
    char string[256 + 1 /*tz*/] = { 0 };

    if (1 == fscanf(file, "%256s", string))
    {
        addToComboBox(box, QString(string));
    }
}

void NetworkDialog::parsePort(FILE *file, QComboBox *box)
{
    int integer = 0;

    if (1 == fscanf(file, "%d", &integer))
    {
        addToComboBox(box, QString("%1").arg(integer));
    }
}

void NetworkDialog::readConfig()
{
    FILE *file = fopen("Comote.config", "r");
    if (file)
    {
        char buf[256 + 1 /*tz*/] = { 0 };

        while (fscanf(file, "%256s", buf) != EOF)
        {
            if (strncmp(buf, "ip", 2) == 0)
            {
                parseHostname(file, ui.comboBoxHostname);
            }
            else if (strncmp(buf, "tcp", 3) == 0)
            {
                parsePort(file, ui.comboBoxTCPPort);
            }
#if USE_TUIO
            else if (strncmp(buf, "udp", 3) == 0)
            {
                parsePort(file, ui.comboBoxUDPPort);
            }
#endif
        }

        fclose(file);
    }

    // Defaults
    //if (ui.comboBoxHostname->count() == 0)
    //{
    //	addToComboBox(ui.comboBoxHostname, "127.0.0.1");
    //}
    if (ui.comboBoxTCPPort->count() == 0)
    {
        addToComboBox(ui.comboBoxTCPPort, "31043");
    }
#if USE_TUIO
    if (ui.comboBoxUDPPort->count() == 0)
    {
        addToComboBox(ui.comboBoxUDPPort, "50096");
    }
#endif
}

void NetworkDialog::writeComboBox(FILE *file, const char *prefix, const QComboBox *box)
{
    if (box->count() > 0)
    {
        // write current item first
        fprintf(file, "%s %s\n", prefix, box->currentText().toStdString().c_str());

        // write all other items
        for (int i = 0; i < box->count(); ++i)
        {
            if (i != box->currentIndex())
            {
                fprintf(file, "%s %s\n", prefix, box->itemText(i).toStdString().c_str());
            }
        }
    }
}

void NetworkDialog::writeConfig()
{
    FILE *file = fopen("Comote.config", "w");
    if (file)
    {
        fprintf(file, "# Comote.config\n\n");

        writeComboBox(file, "ip", ui.comboBoxHostname);
        writeComboBox(file, "tcp", ui.comboBoxTCPPort);
#if USE_TUIO
        writeComboBox(file, "udp", ui.comboBoxUDPPort);
#endif

        fclose(file);
    }
}
