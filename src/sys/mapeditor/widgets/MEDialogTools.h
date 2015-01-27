/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DIALOGTOOLS_H
#define ME_DIALOGTOOLS_H

#include <QDialog>

class QLineEdit;
class QListWidget;

class MERenameDialog : public QDialog
{
    Q_OBJECT

public:
    enum mode
    {
        SINGLE,
        GROUP
    };
    MERenameDialog(int mode, const QString &text, QWidget *parent = 0);

private:
    QLineEdit *m_renameLineEdit;

private slots:
    void rejected();
    void accepted();
};

class MEDeleteHostDialog : public QDialog
{
    Q_OBJECT

public:
    MEDeleteHostDialog(QWidget *parent = 0);
    QString getLine();

private:
    QListWidget *m_deleteHostBox;

private slots:
    void rejected();
    void accepted();
};

class MEMirrorHostDialog : public QDialog
{
    Q_OBJECT

public:
    MEMirrorHostDialog(QWidget *parent = 0);

private slots:
    void selectMirrorHost(int);
};

#endif
