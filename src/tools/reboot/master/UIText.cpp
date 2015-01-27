/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "UIText.h"

#include "RemoteRebootMaster.h"

#include "qtextstream.h"

#include <iostream>
using namespace std;

UIText::UIText(RemoteRebootMaster *master)
    : UI(master)
{
}

UIText::~UIText()
{
}

QString UIText::getUserInput(const QString &name, const QString &type, const QString &defaultValue)
{

    QString input;
    bool inputOk = false;
    QTextIStream instream(stdin);

    while (!inputOk)
    {
        cout << "Enter " << name << " (" << type << ") " << (defaultValue == "" ? defaultValue : "[" + defaultValue + "]") << ": ";
        instream >> input;

        if (input == "")
            input = defaultValue;

        if (type == "string")
        {
            inputOk = true;
        }
        else if (type == "passwd")
        {
            inputOk = true;
        }
        else if (type == "int")
        {
            input.toInt(&inputOk, 10);
        }
    }

    return input;
}

QMap<QString, QString> UIText::getUserInputs(const QStringList &inputList)
{

    QMap<QString, QString> input;

    for (QStringList::const_iterator i = inputList.begin(); i != inputList.end(); ++i)
    {
        QString name = *i;
        ++i;
        QString type = *i;
        ++i;
        QString defaultValue = *i;

        input.insert(name, getUserInput(name, type, defaultValue));
    }

    return input;
}

void UIText::exec()
{

    QStringList items = master->getBootEntries();
    int ctr = 1;
    int selected = 0;

    cout << "************************************************************" << endl;
    cout << " Select boot image:" << endl;
    for (QStringList::iterator i = items.begin(); i != items.end(); ++i)
    {
        cout << " " << ctr << ": " << *i << endl;
        ++ctr;
    }
    cout << "************************************************************" << endl;

    do
    {
        cout << "> ";
        QString input;
        QTextIStream instream(stdin);
        instream >> input;
        selected = input.toInt();
    } while ((selected <= 0) || (selected >= ctr));

    master->setDefaultBoot(--selected);
    master->reboot();
}
