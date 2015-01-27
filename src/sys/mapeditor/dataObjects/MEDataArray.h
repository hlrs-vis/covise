/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DATAARRAY_H
#define ME_DATAARRAY_H

#include <QWidget>

class QVBoxLayout;
class QCheckBox;
class QToolBox;
class QTextEdit;
class QLineEdit;
class QTableWidget;
class QLabel;

class MEPort;
class MEDataObject;
class MEDataArray;
class MEDataViewer;
class MEDataTreeItem;

namespace covise
{
class coRecvBuffer;
class coDoInfo;
class coDistributedObject;
}

//================================================
class MEDataArray : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEDataArray(covise::coRecvBuffer &, MEDataTreeItem *, QWidget *parent = 0);
    MEDataArray(MEDataTreeItem *, QWidget *parent = 0);

    void update();
    QString getName()
    {
        return m_dataname;
    };
    MEDataTreeItem *getItem()
    {
        return m_item;
    };

private:
    covise::coDoInfo *m_dataObjectInfo;
    const covise::coDistributedObject *m_object, *m_tmpObject;

    int *m_dataPointer;
    int m_current, m_start, m_nele, m_ncol, m_step, m_index, m_nrows, m_cols, m_min, m_max;

    QString m_dataname, m_parentname;
    QStringList m_values;
    QTableWidget *m_table;
    QVBoxLayout *main;
    QLineEdit *m_lmin, *m_lmax, *m_lncol, *m_lstep, *m_lindex, *m_lvalue;
    QLabel *m_caption;
    QCheckBox *m_notation;

    MEDataTreeItem *m_item;

    bool getDistributedDataObjectInfo();
    int *getDataObjectPointer(int);
    void makeLayout();
    void updateArray();
    void fillTable(int rows, int cols);

public slots:

    void closeCB();

private slots:

    void startCB();
    void lengthCB();
    void stepCB();
    void widthCB();
    void indexCB();
    void backCB();
    void forwardCB();
    void notationCB(int state);
};
#endif
