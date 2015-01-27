/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COQTFILESELECTION_H
#define COQTFILESELECTION_H

#include <q3hbox.h>
#include <qstring.h>

#include <util/coTypes.h>

class QLineEdit;
class QPushButton;

class CONFIGEDITOREXPORT coQtFileSelection : public Q3HBox
{

    Q_OBJECT

public:
    coQtFileSelection(QString filename = QString::null,
                      QWidget *parent = 0);
    ~coQtFileSelection();

    void setFileName(const QString &filename);
    QString getFileName() const;

signals:
    void fileNameChanged(const QString &filename);

private slots:
    void textChanged(const QString &text);
    void openFileChooser();

private:
    QLineEdit *filenameTF;
    QPushButton *openFCButton;
};

#endif
