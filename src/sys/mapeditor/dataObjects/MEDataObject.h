/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DATAOBJECT_H
#define ME_DATAOBJECT_H

#include <QVector>
#include <QWidget>

class QSplitter;
class QVBoxLayout;
class QGridLayout;
class QGroupBox;
class QTextEdit;
class QLabel;
class QLineEdit;
class QPushButton;

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

class MyPushButton;

//================================================
class MEDataObject : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEDataObject(covise::coRecvBuffer &, MEDataTreeItem *, QWidget *parent = 0);
    MEDataObject(MEDataTreeItem *, QWidget *parent = 0);

    // data types
    enum dataType
    {
        POINTER = 0,
        ARRAY
    };

    QString getName()
    {
        return m_objname;
    };
    MEDataTreeItem *getItem()
    {
        return m_item;
    };
    void update(const QString &);
    bool hasDataObjectInfo()
    {
        return m_hasObjectInfo;
    };

private:
    MEDataTreeItem *m_item;

    covise::coDoInfo *m_dataObjectInfo;
    const covise::coDistributedObject *m_object;

    bool m_hasObjectInfo;
    int *m_dataPointer;
    int m_icount, m_nattributes, m_currentType;
    QString valueBuffer, typeBuffer;

    QLabel *m_type, *m_name, *m_host;
    QString m_objtype, m_objname, m_location;

    QStringList m_attributeNames;
    QStringList m_attributeLabels;

    int m_attrListLength;
    QVector<QLabel *> m_attributeNameList;
    QVector<QLabel *> m_attributeLabelList;
    QVector<QPushButton *> m_attributeMoreList;

    int m_dataListLength;
    QVector<QPushButton *> m_dataNameList;
    QVector<QLabel *> m_dataTypeList;
    QVector<QLabel *> m_dataValueList;
    QVector<MEDataTreeItem *> m_dataItemList;

    void makeLayout();
    void makeLayout(covise::coRecvBuffer &);
    bool getDistributedDataObjectInfo();
    void getDistributedDataObjectInfo(covise::coRecvBuffer &);
    bool getDataObjectPointer(int);
    void updateLayout();
    void makeGeneralInfo(QVBoxLayout *main);
    void makeAttributeInfo(QVBoxLayout *main);
    void makeDataStructureInfo(QGroupBox *gbdata, QGridLayout *grid);
    void makeDataStructureInfo(QGroupBox *gbdata, QGridLayout *grid, covise::coRecvBuffer &);

private slots:

    void moreCB();
    void infoCB();
};
#endif
