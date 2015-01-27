/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QPushButton>
#if QT_VERSION >= 0x040400
#include <QFormLayout>
#endif
#include <QGridLayout>
#include <QLabel>
#include <QGroupBox>
#include <QMessageBox>
#include <QDebug>

#include <do/coDistributedObject.h>
#include "MEDataViewer.h"
#include "MEDataObject.h"
#include "MEDataTree.h"
#include "ports/MEDataPort.h"
#include "handler/MEMainHandler.h"

using covise::coDoInfo;
using covise::coObjInfo;

/*!
    \class MEDataObject
    \brief Widget shows the content and structure of a data object

    This class is part of the MEDataViewer
*/

MEDataObject::MEDataObject(MEDataTreeItem *it, QWidget *p2)
    : QWidget(p2)
    , m_item(it)
    , m_dataObjectInfo(NULL)
    , m_dataPointer(NULL)
{
    // cut portname out of text
    QString tmp = m_item->text(0);
    if (tmp.contains(":: "))
    {
        m_objname = tmp.section(':', -1);
        m_objname = m_objname.remove(0, 1);
    }

    else
        m_objname = tmp;

    m_hasObjectInfo = getDistributedDataObjectInfo();
    if (m_hasObjectInfo)
        makeLayout();
}

//!
//! create the main data info widget
//!
void MEDataObject::makeLayout()
{

    // create the main layout

    QVBoxLayout *main = new QVBoxLayout(this);

    // show general object informations

    makeGeneralInfo(main);

    // show attributes

    if (m_nattributes > 0)
        makeAttributeInfo(main);

    // add the data information part

    QGroupBox *gbdata = new QGroupBox("Data");
    gbdata->setFont(MEMainHandler::s_boldFont);
    gbdata->setAlignment(Qt::AlignLeft);

    QGridLayout *grid = new QGridLayout();
    gbdata->setLayout(grid);
    grid->setColumnStretch(0, 0);
    grid->setColumnStretch(1, 0);
    grid->setColumnStretch(2, 1);
    grid->setHorizontalSpacing(20);

    m_dataListLength = m_icount;
    makeDataStructureInfo(gbdata, grid);

    main->addWidget(gbdata);
    main->addStretch(5);
    show();
}

#define addLine(label2, text1, text2, pos)        \
    label1 = new QLabel(text1, gb);               \
    label1->setFont(MEMainHandler::s_italicFont); \
    grid->addWidget(label1, pos, 0);              \
    label2 = new QLabel(text2, gb);               \
    grid->addWidget(label2, pos, 1);

//!
//! upper part of data info, general object information
//!
void MEDataObject::makeGeneralInfo(QVBoxLayout *main)
{

    // create a groupbox with a grid layout inside

    QGroupBox *gb = new QGroupBox("General information");
    gb->setFont(MEMainHandler::s_boldFont);
    gb->setAlignment(Qt::AlignLeft);

    QGridLayout *grid = new QGridLayout();
    gb->setLayout(grid);
    grid->setColumnStretch(0, 0);
    grid->setColumnStretch(1, 1);
    grid->setHorizontalSpacing(5);

    // create content

    QLabel *label1;
    addLine(m_type, "Type : ", m_objtype, 0);
    addLine(m_name, "Name : ", m_objname, 1);
    addLine(m_host, "Host : ", m_location, 2);

    main->addWidget(gb);
}

//!
//! show attributes of an object
//!
void MEDataObject::makeAttributeInfo(QVBoxLayout *main)
{

    // create a group box with a gridlayout inside

    QGroupBox *gb = new QGroupBox("Attributes");
    gb->setAlignment(Qt::AlignLeft);
    gb->setFont(MEMainHandler::s_boldFont);

    QGridLayout *grid = new QGridLayout();
    gb->setLayout(grid);
    grid->setColumnStretch(0, 0);
    grid->setColumnStretch(1, 0);
    grid->setColumnStretch(2, 1);
    grid->setHorizontalSpacing(10);

    m_attrListLength = m_nattributes;

    // create widgets for attributes

    int maxSize = 25;
    for (int i = 0; i < m_nattributes; i++)
    {
        QLabel *ll = new QLabel(m_attributeNames[i] + " :  ", gb);
        ll->setFont(MEMainHandler::s_italicFont);
        grid->addWidget(ll, i, 0, Qt::AlignLeft);
        m_attributeNameList << ll;

        // attributes can be very long and seperated by \n
        // show only the first item and not more than 25 characters
        QString sub = m_attributeLabels[i].section("\n", 0, 0);
        QString str = sub.left(maxSize);
        QLabel *line = new QLabel(str, gb);
        m_attributeLabelList << line;
        grid->addWidget(line, i, 1, Qt::AlignLeft);

        // if attribute label is too long show a separate popup widget
        QPushButton *pb = new QPushButton(gb);
        pb->setText("more...");
        pb->setToolTip(m_attributeLabels[i]);
        m_attributeMoreList << pb;
        grid->addWidget(pb, i, 2, Qt::AlignLeft);
        connect(pb, SIGNAL(clicked()), this, SLOT(moreCB()));

        if (m_attributeLabels[i].contains("\n") || m_attributeLabels[i].size() > maxSize)
            pb->show();
        else
            pb->hide();
    }

    main->addWidget(gb);
}

//!
//! update the information for a data object
//!
void MEDataObject::update(const QString &text)
{
    //m_item->adaptFont();
    m_objname = text;

    m_object = covise::coDistributedObject::createFromShm(coObjInfo(m_objname.toLatin1().data()));

    // get all infos
    if (m_object != NULL)
    {
        // data object type
        m_objtype = MEDataPort::getDataObjectString(m_object, false);

        // ip address
        m_location = m_object->object_on_hosts();
        if (m_location.isEmpty())
            m_location = "LOCAL";

        // all attributes
        const char **names, **labels;
        m_nattributes = m_object->getAllAttributes(&names, &labels);
        m_attributeNames.clear();
        m_attributeLabels.clear();
        for (int i = 0; i < m_nattributes; i++)
        {
            m_attributeNames += QString(names[i]);
            m_attributeLabels += QString(labels[i]);
        }

        // info about shm memory
        m_icount = m_object->getObjectInfo(&m_dataObjectInfo);

        // create the layout
        updateLayout();
    }
}

//!
//! update the layout when a module was executed again
//!
void MEDataObject::updateLayout()
{

    int iend;

    // update the general information part
    if (m_type == NULL)
    {
        //object was not ok before but now, so we have to make the layout this time.
        makeLayout();
    }

    m_type->setText(m_objtype);
    m_name->setText(m_objname);
    m_host->setText(m_location);

    // update the attribute information part

    if (m_nattributes > m_attrListLength)
        iend = m_attrListLength;
    else
        iend = m_nattributes;

    for (int i = 0; i < iend; i++)
    {
        m_attributeNameList.at(i)->setText(m_attributeNames[i]);
        m_attributeLabelList.at(i)->setText(m_attributeLabels[i]);

        // if attribute label is too long make a separate popup widget
        if (!m_attributeMoreList.isEmpty())
        {
            QPushButton *pb = m_attributeMoreList.at(i);
            if (m_attributeLabels[i].length() > 25)
                pb->show();

            else
                pb->hide();
        }
    }

    // add the data structure information part

    if (m_icount > m_dataListLength)
        iend = m_dataListLength;
    else
        iend = m_icount;

    for (int i = 0; i < iend; i++)
    {
        bool haveChildren = getDataObjectPointer(i);

        // description
        if (m_dataObjectInfo[i].obj_name != NULL)
        {
            m_dataNameList.at(i)->setText(m_dataObjectInfo[i].obj_name);
            m_dataItemList.at(i)->setText(0, m_dataObjectInfo[i].obj_name);
            m_dataItemList.at(i)->updateItem();
        }

        else
        {
            if (haveChildren)
            {
                m_dataNameList.at(i)->setText(m_dataObjectInfo[i].description);
                m_dataItemList.at(i)->setText(0, m_dataObjectInfo[i].description);
                m_dataItemList.at(i)->updateItem();
            }

            else
                m_dataNameList.at(i)->setText(m_dataObjectInfo[i].description);
        }

        // type
        m_dataTypeList.at(i)->setText(typeBuffer);

        // value
        m_dataValueList.at(i)->setText(valueBuffer);
    }
}

//!
//! get the data type of the data object
//!
bool MEDataObject::getDataObjectPointer(int i)
{
    bool haveChildren = false;
    switch (m_dataObjectInfo[i].type)
    {
    case covise::CHARSHM:
        typeBuffer.sprintf("%c", *(char *)m_dataObjectInfo[i].ptr);
        valueBuffer.sprintf("char");
        break;
    case covise::SHORTSHM:
        typeBuffer.sprintf("%d", *(short *)m_dataObjectInfo[i].ptr);
        valueBuffer.sprintf("short");
        break;
    case covise::INTSHM:
        typeBuffer.sprintf("%d", *(int *)m_dataObjectInfo[i].ptr);
        valueBuffer.sprintf("int");
        break;
    case covise::LONGSHM:
        typeBuffer.sprintf("%ld", *(long *)m_dataObjectInfo[i].ptr);
        valueBuffer.sprintf("long");
        break;
    case covise::FLOATSHM:
        typeBuffer.sprintf("%#f", *(float *)m_dataObjectInfo[i].ptr);
        valueBuffer.sprintf("float");
        break;
    case covise::DOUBLESHM:
        typeBuffer.sprintf("%#E", *(double *)m_dataObjectInfo[i].ptr);
        valueBuffer.sprintf("double");
        break;
    case covise::CHARSHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length (-1 for 64-bit workaround)
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("char[]");
        haveChildren = 1;
        break;
    case covise::SHORTSHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("short[]");
        haveChildren = true;
        break;
    case covise::INTSHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("int[]");
        haveChildren = true;
        break;
    case covise::LONGSHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("long[]");
        haveChildren = true;
        break;
    case covise::FLOATSHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("float[]");
        haveChildren = true;
        break;
    case covise::DOUBLESHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("double[]");
        haveChildren = true;
        break;
    case covise::STRINGSHMARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("string[]");
        haveChildren = true;
        break;
    case covise::SHMPTRARRAY:
        m_dataPointer = (int *)m_dataObjectInfo[i].ptr;
        // length
        typeBuffer.sprintf("%d", m_dataPointer[1] - 1);
        valueBuffer.sprintf("shm[]");
        haveChildren = true;
        break;
    default:
        typeBuffer.sprintf("%s", m_dataObjectInfo[i].type_name);
        valueBuffer.sprintf("shm[]");
        break;
    }
    return haveChildren;
}

//!
//! show the internal data structure
//!
bool MEDataObject::getDistributedDataObjectInfo()
{
    bool ok = false;

    m_object = covise::coDistributedObject::createFromShm(coObjInfo(m_objname.toLatin1().data()));

    // get all infos
    if (m_object != NULL)
    {
        // data object type
        m_objtype = MEDataPort::getDataObjectString(m_object, false);

        // ip address
        m_location = m_object->object_on_hosts();
        if (m_location.isEmpty())
            m_location = "LOCAL";

        const char **names, **labels;
        m_nattributes = m_object->getAllAttributes(&names, &labels);
        for (int i = 0; i < m_nattributes; i++)
        {
            m_attributeNames += QString(names[i]);
            m_attributeLabels += QString(labels[i]);
        }

        // info about shm memory
        m_icount = m_object->getObjectInfo(&m_dataObjectInfo);
        ok = true;
    }
    return ok;
}

//!
//! create the main data info widget
//!
void MEDataObject::makeDataStructureInfo(QGroupBox *gbdata, QGridLayout *grid)
{

    for (int i = 0; i < m_icount; i++)
    {
        int open = getDataObjectPointer(i);

        // description
        if (m_dataObjectInfo[i].obj_name != NULL)
        {
            m_currentType = POINTER;
            QPushButton *l = new QPushButton(m_dataObjectInfo[i].obj_name, gbdata);
            connect(l, SIGNAL(clicked()), this, SLOT(infoCB()));

            MEDataTreeItem *it = new MEDataTreeItem(m_item, m_dataObjectInfo[i].obj_name);
            it->setObjType(m_currentType);
            it->setIcon(0, MEMainHandler::instance()->pm_folderclosed);

            // create dummy object
            new MEDataTreeItem(it, "dummy");

            m_dataItemList.append(it);
            m_dataNameList.append(l);
            grid->addWidget(l, i, 0, Qt::AlignLeft);
        }

        else
        {
            if (open)
            {
                m_currentType = ARRAY;
                QPushButton *l = new QPushButton(m_dataObjectInfo[i].description, gbdata);
                connect(l, SIGNAL(clicked()), this, SLOT(infoCB()));

                MEDataTreeItem *it = new MEDataTreeItem(m_item, m_dataObjectInfo[i].description);
                it->setObjType(m_currentType);
                it->setIcon(0, MEMainHandler::instance()->pm_table);

                m_dataItemList.append(it);
                m_dataNameList.append(l);
                grid->addWidget(l, i, 0, Qt::AlignLeft);
            }

            else
            {
                QPushButton *l = new QPushButton(m_dataObjectInfo[i].description, gbdata);
                l->setFont(MEMainHandler::s_italicFont);
                l->setFlat(true);
                l->setAutoFillBackground(true);

                m_dataNameList.append(l);
                m_dataItemList.append(NULL);
                grid->addWidget(l, i, 0, Qt::AlignLeft);
            }
        }

        // type
        QLabel *l = new QLabel(typeBuffer, gbdata);
        m_dataTypeList.append(l);
        grid->addWidget(l, i, 1, Qt::AlignLeft);

        // value
        l = new QLabel(valueBuffer, gbdata);
        m_dataValueList.append(l);
        grid->addWidget(l, i, 2, Qt::AlignLeft);
    }
}

//!
//! popup a window for more information about a data attribute
//!
void MEDataObject::moreCB()
{
    const QObject *o = QObject::sender();
    if (const QPushButton *pb = qobject_cast<const QPushButton *>(o))
    {
        int index = m_attributeMoreList.indexOf(const_cast<QPushButton *>(pb));
        if (index >= 0)
            QMessageBox::information(const_cast<QPushButton *>(pb), "COVISE Data Viewer", m_attributeLabels[index]);
        else
            qWarning("QPushButton not found");
    }
    else
        qWarning("Sender QObject is not a QPushButton");
}

//!
//! show more info about pressed data object
//!
void MEDataObject::infoCB()
{
    const QObject *o = QObject::sender();
    QPushButton *pb = (QPushButton *)o;

    int index = m_dataNameList.indexOf(pb);
    if (index != -1)
    {
        MEDataTreeItem *it = m_dataItemList.at(index);
        it->showItemContent();
    }
}
