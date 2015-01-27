/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QTableWidget>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include <do/coDistributedObject.h>
#ifdef YAC
#include "yac/coQTSendBuffer.h"
#endif

#include "MEDataArray.h"
#include "MEDataViewer.h"
#include "MEDataTree.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "widgets/MEUserInterface.h"

using covise::coDoInfo;

/*!
    \class MEDataArray
    \brief Widget displays a data object array

    Three arrays can be displayed at the same time <br>
    This class is part of the MEDataViewer
*/

#ifdef YAC

MEDataArray::MEDataArray(covise::coRecvBuffer &tb, MEDataTreeItem *it, QWidget *p2)
    : QWidget(p2)
    , m_table(NULL)
    , m_item(it)
{

    // get values
    m_current = m_item->getIndex();
    tb >> m_min >> m_max;
    const char *s;
    int len = m_max - m_min + 1;
    for (int i = 0; i < len; i++)
    {
        tb >> s;
        m_values << s;
    }

    m_dataname = m_item->text(0);
    m_parentname = (m_item->parent())->parent()->text(0);
    makeLayout();
    hide();
}

#else

MEDataArray::MEDataArray(MEDataTreeItem *it, QWidget *p2)
    : QWidget(p2)
    , m_dataObjectInfo(NULL)
    , m_object(NULL)
    , m_tmpObject(NULL)
    , m_dataPointer(NULL)
    , m_current(0)
    , m_table(NULL)
    , m_item(it)
{

    // cut portname out of text
    QString tmp = m_item->text(0);
    if (tmp.contains(":: "))
    {
        m_dataname = tmp.section(':', -1);
        m_dataname = m_dataname.remove(0, 1);
    }

    else
        m_dataname = tmp;

    tmp = m_item->parent()->text(0);
    if (tmp.contains(":: "))
    {
        m_parentname = tmp.section(':', -1);
        m_parentname = m_parentname.remove(0, 1);
    }

    else
        m_parentname = tmp;

    // get distributed data object info
    if (getDistributedDataObjectInfo())
        makeLayout();

    hide();
}
#endif

#define addLine(pb, text, text2, callback, tip)                 \
    hbox->addWidget(new QLabel(text), 0);                       \
    pb = new QLineEdit(text2);                                  \
    hbox->addWidget(pb, 1);                                     \
    pb->setToolTip(tip);                                        \
    connect(pb, SIGNAL(returnPressed()), this, SLOT(callback)); \
    connect(pb, SIGNAL(editingFinished()), this, SLOT(callback));

//!
//! create two info textlines & a table for data
//!
void MEDataArray::makeLayout()
{

// init some variable

#ifdef YAC
    m_start = m_min;
    m_nele = m_max - m_min + 1;
#else
    m_start = 0;
    m_nele = qMin(200, m_dataPointer[1] - 1);
#endif
    m_step = 1;
    m_ncol = 10;

    // create the main layout

    main = new QVBoxLayout(this);
    main->setMargin(2);
    main->setSpacing(2);

    // add a title line and use the right hist color

    QFrame *frame = new QFrame(this);
    QHBoxLayout *hbox = new QHBoxLayout(frame);

    QColor color = m_item->getColor();
    QPalette palette;
    palette.setBrush(backgroundRole(), color);
    frame->setPalette(palette);
    frame->setAutoFillBackground(true);
    frame->setFrameStyle(QFrame::Panel | QFrame::Raised);

    // close button

    QPushButton *bb = new QPushButton();
    bb->setIcon(MEMainHandler::instance()->pm_stop);
    bb->setFlat(true);
    bb->setToolTip("Close these widget");
    connect(bb, SIGNAL(clicked()), this, SLOT(closeCB()));
    hbox->addWidget(bb, 0);

    // data object name

    m_caption = new QLabel(m_parentname + "->" + m_dataname, this);
    hbox->addWidget(m_caption, 0);

    // add spacer widget

    hbox->addStretch(3);

    // widgets for showing a specific table item (data element)

    hbox->addWidget(new QLabel("Value[i]"), 0);

    m_lindex = new QLineEdit();
    m_lindex->setToolTip("Shows the content of a given data element");
    connect(m_lindex, SIGNAL(returnPressed()), this, SLOT(indexCB()));
    hbox->addWidget(m_lindex, 0);

    m_lvalue = new QLineEdit();
    m_lvalue->setFrame(false);
    m_lvalue->setReadOnly(true);
    m_lvalue->setToolTip("Shows the content of a given data element");
    hbox->addWidget(m_lvalue, 0);

    // button for hex notation

    m_notation = new QCheckBox();
    m_notation->setText("Hex");
    m_notation->setCheckState(Qt::Unchecked);
    m_notation->setToolTip("Shows the data elements in hexadecimal notation");
    connect(m_notation, SIGNAL(stateChanged(int)), this, SLOT(notationCB(int)));
    hbox->addWidget(m_notation, 0);

    main->addWidget(frame);

    // add a info line for modifing the content m_table

    QWidget *w = new QWidget(this);

    //QGridLayout *hbx = new QGridLayout();
    hbox = new QHBoxLayout(w);

    addLine(m_lmin, "Start", QString::number(m_start), startCB(), "First data element");
    addLine(m_lmax, "Length", QString::number(m_nele), lengthCB(), "Number of shown elements");
    addLine(m_lstep, "Step", QString::number(m_step), stepCB(), "Stepwidth");
    addLine(m_lncol, "TableWidth", QString::number(m_ncol), widthCB(), "Number of table columns");

    bb = new QPushButton();
    bb->setIcon(QPixmap(":/icons/2leftarrow.png"));
    bb->setToolTip("Shows former data elemnts");
    connect(bb, SIGNAL(clicked()), this, SLOT(backCB()));
    hbox->addWidget(bb, 0);

    bb = new QPushButton();
    bb->setIcon(QPixmap(":/icons/2rightarrow.png"));
    bb->setToolTip("Shows next elements");
    connect(bb, SIGNAL(clicked()), this, SLOT(forwardCB()));
    hbox->addWidget(bb, 0);

    main->addWidget(w);

    // make the m_table content visible

    updateArray();
    //setFixedWidth(this->sizeHint().width());
    show();
}

//!
//! update the content of one array
//!
void MEDataArray::updateArray()
{

    if (m_nele == 0)
        return;

    // init defaults

    int cols = m_ncol;
    int nanz = m_nele;
    int nrows = (nanz + cols - 1) / cols;
    //nrows     = qMin(nrows, 50);
    nrows = nanz / cols + 1;
    nrows = qMax(nrows, 1);

    // reset table

    if (m_table)
        delete m_table;

    m_table = new QTableWidget(nrows, cols, this);
    m_lmax->setText(QString::number(m_nele));
    m_lmin->setText(QString::number(m_start));

    // create horizontal header

    QStringList hheader;
    for (int i = 0; i < cols; i++)
    {
        int k = i * m_step;
        hheader << QString::number(k);
    }

    // create vertical header

    int curr = m_start;
    QStringList vheader;
    for (int i = 0; i < nrows; i++)
    {
        vheader << QString::number(curr);
        for (int j = 0; j < cols; j++)
            curr = curr + m_step;
    }

    // fill with elements

    fillTable(nrows, cols);

    m_table->setHorizontalHeaderLabels(hheader);
    m_table->setVerticalHeaderLabels(vheader);
    m_table->show();

    main->addWidget(m_table, 1);
}

//!
//! get all necessary array information for a distributed data object
//!
bool MEDataArray::getDistributedDataObjectInfo()
{
#ifdef YAC

    return false;

#else

    // check parent dataobject

    m_object = covise::coDistributedObject::createFromShm(covise::coObjInfo(m_parentname.toLatin1().data()));

    // get information about shared memory
    if (m_object != NULL)
    {
        int icount = m_object->getObjectInfo(&m_dataObjectInfo);

        // search for this data object in the information gathered from the parent
        for (int i = 0; i < icount; i++)
        {
            m_dataPointer = getDataObjectPointer(i);

            // description
            if (m_dataObjectInfo[i].obj_name == NULL && m_dataPointer && QString::compare(m_dataObjectInfo[i].description, m_dataname) == 0)
            {
                m_current = i;
                return true;
            }
        }
    }
    return false;
#endif
}

//!
//! get the pointer to data object
//!
int *MEDataArray::getDataObjectPointer(int i)
{

#ifdef YAC

    Q_UNUSED(i)
    return 0;

#else

    int *pointer = NULL;
    switch (m_dataObjectInfo[i].type)
    {
    case covise::CHARSHMARRAY:
    case covise::SHORTSHMARRAY:
    case covise::INTSHMARRAY:
    case covise::LONGSHMARRAY:
    case covise::FLOATSHMARRAY:
    case covise::DOUBLESHMARRAY:
    case covise::STRINGSHMARRAY:
    case covise::SHMPTRARRAY:
        pointer = (int *)m_dataObjectInfo[i].ptr;
        break;
    }

    return pointer;
#endif
}

//!
//! print array elements
//!
void MEDataArray::fillTable(int rows, int cols)
{

// loop over data

#ifdef YAC

    int curr = m_start;
    int end = m_values.count();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (curr >= end)
                break;
            m_table->setItem(i, j, new QTableWidgetItem(m_values[curr]));
            curr += m_step;
        }
    }

#else

    QString buffer;
    if (m_notation->isChecked()
        && m_dataObjectInfo[m_current].type != covise::CHARSHMARRAY
        && m_dataObjectInfo[m_current].type != covise::STRINGSHMARRAY)
    {
        int *cur = &m_dataPointer[2];
        int *end = cur + m_dataPointer[1] - 1;
        cur += m_start;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (cur >= end)
                    break;
                buffer.sprintf("%08x", *cur);
                m_table->setItem(i, j, new QTableWidgetItem(buffer));
                cur += m_step;
            }
        }
    }

    else
    {
        switch (m_dataObjectInfo[m_current].type)
        {
        case covise::CHARSHMARRAY:
        case covise::STRINGSHMARRAY:
        {
            const char *format = "'%c' %d";
            if (m_notation->isChecked())
                format = "'%c' %02x";
            unsigned char *cur = reinterpret_cast<unsigned char *>(&m_dataPointer[2]);
            unsigned char *end = cur + m_dataPointer[1] - 1;
            cur += m_start;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (cur >= end)
                        break;
                    buffer.sprintf(format, *cur, static_cast<int>(*cur));
                    m_table->setItem(i, j, new QTableWidgetItem(buffer));
                    cur += m_step;
                }
            }
            break;
        }

        case covise::INTSHMARRAY:
        {
            int *cur = reinterpret_cast<int *>(&m_dataPointer[2]);
            int *end = cur + m_dataPointer[1] - 1;
            cur += m_start;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (cur >= end)
                        break;
                    m_table->setItem(i, j, new QTableWidgetItem(QString::number(*cur)));
                    cur += m_step;
                }
            }
            break;
        }

        case covise::FLOATSHMARRAY:
        {
            float *cur = reinterpret_cast<float *>(&m_dataPointer[2]);
            float *end = cur + m_dataPointer[1] - 1;
            cur += m_start;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (cur >= end)
                        break;
                    m_table->setItem(i, j, new QTableWidgetItem(QString::number(*cur)));
                    cur += m_step;
                }
            }
            break;
        }

        default:
        {
            MEUserInterface::instance()->printMessage("COVISE Data Viewer:: Array of this datatype can't be printed");
            break;
        }
        }
    }
#endif

    m_table->resizeColumnsToContents();
    m_table->resizeRowsToContents();
}

//!
//! update content of a data array
//!
void MEDataArray::update()
{

    // init variables
    m_dataname = m_item->text(0);
    m_parentname = m_item->QTreeWidgetItem::parent()->text(0);
    m_current = 0;
    m_caption->setText(m_parentname + "->" + m_dataname);

    // get distributed data object info
    if (getDistributedDataObjectInfo())
        updateArray();
}

//!
//! close the window and remove it from splitter
//!
void MEDataArray::closeCB()
{
    setParent(0);
    hide();
}

//!
//! callback, a new start element was set
//!
void MEDataArray::startCB()
{

    int nmin = m_lmin->text().toInt();

#ifdef YAC

    if (nmin < m_start)
    {
        m_start = nmin;
        covise::coSendBuffer sb;
        int i1 = getItem()->parent()->text(1).toInt();
        int i2 = getItem()->parent()->text(2).toInt();
        int i3 = getItem()->parent()->text(3).toInt();
        sb << i1 << i2 << i3 << getItem()->getIndex() << m_start << m_nele - m_start + 1;
        MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_GET_OBJ_ARRAY, sb);
    }

    else
    {
        m_start = nmin;
        updateArray();
    }

#else

    m_start = qMin(nmin, m_dataPointer[1] - 1);
    m_nele = qMin(m_nele, m_start + m_dataPointer[1] - 1);
    updateArray();
#endif
}

//!
//! callback, a new maximum array range was set
//!
void MEDataArray::lengthCB()
{

    int nmax = m_lmax->text().toInt();

#ifdef YAC

    if (nmax > m_nele)
    {
        m_nele = nmax;
        covise::coSendBuffer sb;
        int i1 = getItem()->parent()->text(1).toInt();
        int i2 = getItem()->parent()->text(2).toInt();
        int i3 = getItem()->parent()->text(3).toInt();
        sb << i1 << i2 << i3 << getItem()->getIndex() << m_start << m_nele - m_start + 1;
        MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_GET_OBJ_ARRAY, sb);
    }
    else
    {
        m_nele = nmax;
        updateArray();
    }

#else

    m_nele = qMin(nmax, m_start + m_dataPointer[1] - 1);
    updateArray();
#endif
}

//!
//! callback, forward icon was pressed
//!
void MEDataArray::forwardCB()
{
    m_start = m_start + m_nele;
    m_start = qMin(m_start, m_dataPointer[1] - 1);
    m_nele = qMin(m_nele, m_start + m_dataPointer[1] - 1);
    updateArray();
}

//!
//! callback, back  icon was pressed
//!
void MEDataArray::backCB()
{
    m_start = m_start - m_nele;
    m_start = qMax(0, m_start);
    updateArray();
}

//!
//! callback, a new step width was set
//!
void MEDataArray::stepCB()
{
    m_step = m_lstep->text().toInt();
    updateArray();
}

//!
//! callback, a new no. of shown columns was set
//!
void MEDataArray::widthCB()
{
    m_ncol = m_lncol->text().toInt();
    updateArray();
}

//!
//! callback, show array in hex notation
//!
void MEDataArray::notationCB(int)
{
    updateArray();
}

//!
//! callback, show the content of a specific cell
//!
void MEDataArray::indexCB()
{

    // get selected m_item & clear old selected table items
    m_index = m_lindex->text().toInt();
    m_table->clearSelection();

#ifdef YAC

    m_lvalue->setText(m_values[m_index]);

#else

    QString buf;

    if (m_notation->isChecked())
    {
        char *tmpchar = (char *)&m_dataPointer[2];
        buf.sprintf("%08x", tmpchar[m_index]);
        m_lvalue->setText(buf);
    }

    else
    {
        switch (m_dataObjectInfo[m_current].type)
        {
        case covise::CHARSHMARRAY:
        case covise::STRINGSHMARRAY:
        {
            char *tmpchar = (char *)&m_dataPointer[2];
            buf.sprintf("%08x", tmpchar[m_index]);
            m_lvalue->setText(buf);
        }
        break;

        case covise::INTSHMARRAY:
        {
            int *tmpint = (int *)&m_dataPointer[2];
            m_lvalue->setText(QString::number(tmpint[m_index]));
        }
        break;

        case covise::FLOATSHMARRAY:
        {
            float *tmpfloat = (float *)&m_dataPointer[2];
            m_lvalue->setText(QString::number(tmpfloat[m_index]));
        }
        break;
        }
    }

    // show item inside table if possible
    if (m_step == 1 && m_index >= m_start && m_index <= m_start + m_nele - 1)
    {
        int row = (m_index - m_start) / m_table->columnCount();
        int col = m_index - m_start - (row * m_table->columnCount());
        QTableWidgetItem *it = m_table->item(row, col);
        it->setSelected(true);
        m_table->scrollToItem(it);
    }
#endif
}
