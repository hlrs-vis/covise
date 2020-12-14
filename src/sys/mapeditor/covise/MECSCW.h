/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_CSCW_H
#define ME_CSCW_H

#include <QDialog>
#include <QStringList>

class QLineEdit;
class QComboBox;
class QString;
class QGridLayout;
class QListWidgetItem;
class QListWidget;

//================================================
class MECSCW : public QDialog
//================================================
{
    Q_OBJECT

public:
    MECSCW(QWidget *parent = 0);
    ~MECSCW();
    void setVrbPartner(const QStringList &vrbPartner);
public slots:

    void accepted();

private:
    QLineEdit *m_selectionHost = nullptr;
    QListWidget *m_listbox = nullptr;
    QStringList m_configHosts;
private slots:

    void setHostCB(QListWidgetItem *);
    void accepted2(QListWidgetItem *);
    void rejected();
};

//================================================
class MECSCWParam : public QDialog
//================================================
{
    Q_OBJECT

public:
    MECSCWParam(QWidget *parent = 0);
    ~MECSCWParam();

    QLineEdit *hostname, *username, *passwd, *display, *timeout;
    QString ipname;
    QComboBox *connectionModes;

    void setMode(int);
    void setTimeout(QString);
    void setHost(QString);
    void setUser(QString);
    void setDefaults(QString);

public slots:

    void accepted();

private slots:

    void rejected();
};
#endif
