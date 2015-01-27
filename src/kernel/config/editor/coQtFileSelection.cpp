/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coQtFileSelection.h"
//#include "coQtFileSelection.moc"

#include <q3filedialog.h>
#include <qlineedit.h>
#include <qpixmap.h>
#include <qpushbutton.h>

#include "coConfigIcons.h"

coQtFileSelection::coQtFileSelection(QString filename, QWidget *parent)
    : Q3HBox(parent)
{

    filenameTF = new QLineEdit(filename, this);
    openFCButton = new QPushButton(this);
    openFCButton->setIconSet(QPixmap(qembed_findImage("fileopen")));

    connect(openFCButton, SIGNAL(clicked()),
            this, SLOT(openFileChooser()));

    connect(filenameTF, SIGNAL(textChanged(const QString &)),
            this, SLOT(textChanged(const QString &)));
}

coQtFileSelection::~coQtFileSelection() {}

void coQtFileSelection::textChanged(const QString &text)
{
    emit fileNameChanged(text);
}

void coQtFileSelection::openFileChooser()
{

    QString filename = Q3FileDialog::getOpenFileName(QDir::homeDirPath());
    if (!filename.isNull())
    {
        filenameTF->setText(filename);
    }
}

void coQtFileSelection::setFileName(const QString &filename)
{
    filenameTF->setText(filename);
}

QString coQtFileSelection::getFileName() const
{
    return filenameTF->text();
}
