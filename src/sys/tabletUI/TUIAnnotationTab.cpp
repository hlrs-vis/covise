/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIAnnotationTab.h"

#include <sstream>
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include "wce_msg.h"
#endif

//*************************************************************
//*****                 Constructor / Deconstructor       *****
//*************************************************************
TUIAnnotationTab::TUIAnnotationTab(int id, int type, QWidget *w, int parent, QString name)
    : TUITab(id, type, w, parent, name)
{
    frame = new QFrame(w);
    frame->setFrameStyle(QFrame::NoFrame);
    widget = frame;

    gridLayout = new QGridLayout(frame);
    gridLayout->setSpacing(6);
    gridLayout->setMargin(9);
    gridLayout->setObjectName(QString::fromUtf8("gridLayout"));

    groupBoxCurrent = new QGroupBox(w);
    groupBoxCurrent->setObjectName(QString::fromUtf8("groupBoxCurrent"));
    QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(7), static_cast<QSizePolicy::Policy>(7));
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(4);
    sizePolicy.setHeightForWidth(groupBoxCurrent->sizePolicy().hasHeightForWidth());
    groupBoxCurrent->setSizePolicy(sizePolicy);
#ifndef _WIN32_WCE
    groupBoxCurrent->setMinimumSize(QSize(200, 320));
#endif

    //layoutWidget = new QWidget(groupBoxCurrent);
    //layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
    //layoutWidget->setGeometry(QRect(10, 20, 241, 331));
    //gridLayout1 = new QGridLayout(layoutWidget);

    gridLayout1 = new QGridLayout(groupBoxCurrent);
    gridLayout1->setSpacing(6);
    gridLayout1->setMargin(6);
    gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));

    sliderColor = new QSlider(groupBoxCurrent);
    sliderColor->setObjectName(QString::fromUtf8("sliderColor"));
    sliderColor->setOrientation(Qt::Horizontal);
    gridLayout1->addWidget(sliderColor, 2, 1, 1, 2);

    checkBoxShowHide = new QCheckBox(groupBoxCurrent);
    checkBoxShowHide->setObjectName(QString::fromUtf8("checkBoxShowHide"));
    checkBoxShowHide->setChecked(true);
    gridLayout1->addWidget(checkBoxShowHide, 0, 0, 1, 2);

    labelScale = new QLabel(groupBoxCurrent);
    labelScale->setObjectName(QString::fromUtf8("labelScale"));
    gridLayout1->addWidget(labelScale, 1, 0, 1, 1);

    pushButtonDelete = new QPushButton(groupBoxCurrent);
    pushButtonDelete->setObjectName(QString::fromUtf8("pushButtonDelete"));
    gridLayout1->addWidget(pushButtonDelete, 4, 2, 1, 1);

    labelColor = new QLabel(groupBoxCurrent);
    labelColor->setObjectName(QString::fromUtf8("labelColor"));
    gridLayout1->addWidget(labelColor, 2, 0, 1, 1);

    sliderScale = new QSlider(groupBoxCurrent);
    sliderScale->setObjectName(QString::fromUtf8("sliderScale"));
    sliderScale->setOrientation(Qt::Horizontal);
    gridLayout1->addWidget(sliderScale, 1, 1, 1, 2);

    pushButtonOk = new QPushButton(groupBoxCurrent);
    pushButtonOk->setObjectName(QString::fromUtf8("pushButtonOk"));
    gridLayout1->addWidget(pushButtonOk, 4, 0, 1, 2);

    spacerItem = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    gridLayout1->addItem(spacerItem, 5, 1, 1, 1);

    textEdit = new QTextEdit(groupBoxCurrent);
    //textEdit = new TUITextCheck(groupBoxCurrent);
    textEdit->setObjectName(QString::fromUtf8("textEdit"));
    QSizePolicy sizePolicy1(static_cast<QSizePolicy::Policy>(7), static_cast<QSizePolicy::Policy>(4));
    sizePolicy1.setHorizontalStretch(1);
    sizePolicy1.setVerticalStretch(1);
    sizePolicy1.setHeightForWidth(textEdit->sizePolicy().hasHeightForWidth());
    textEdit->setSizePolicy(sizePolicy1);
#ifndef _WIN32_WCE
    textEdit->setMinimumSize(QSize(100, 50));
#endif
    gridLayout1->addWidget(textEdit, 3, 0, 1, 3);

    gridLayout->addWidget(groupBoxCurrent, 1, 1, 1, 1);

    groupBoxAll = new QGroupBox(w);
    groupBoxAll->setObjectName(QString::fromUtf8("groupBoxAll"));
    QSizePolicy sizePolicy2(static_cast<QSizePolicy::Policy>(5), static_cast<QSizePolicy::Policy>(5));
    sizePolicy2.setHorizontalStretch(0);
    sizePolicy2.setVerticalStretch(2);
    sizePolicy2.setHeightForWidth(groupBoxAll->sizePolicy().hasHeightForWidth());
    groupBoxAll->setSizePolicy(sizePolicy2);
#ifndef _WIN32_WCE
    groupBoxAll->setMinimumSize(QSize(260, 160));
#endif

    //layoutWidget1 = new QWidget(groupBoxAll);
    //layoutWidget1->setObjectName(QString::fromUtf8("layoutWidget1"));
    //layoutWidget1->setGeometry(QRect(10, 20, 241, 161));
    //gridLayout2 = new QGridLayout(layoutWidget1);

    gridLayout2 = new QGridLayout(groupBoxAll);
    gridLayout2->setSpacing(6);
    gridLayout2->setMargin(6);
    gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));

    pushButtonDeleteAll = new QPushButton(groupBoxAll);
    pushButtonDeleteAll->setObjectName(QString::fromUtf8("pushButtonDeleteAll"));
    gridLayout2->addWidget(pushButtonDeleteAll, 4, 1, 1, 1);

    sliderScaleAll = new QSlider(groupBoxAll);
    sliderScaleAll->setObjectName(QString::fromUtf8("sliderScaleAll"));
    sliderScaleAll->setOrientation(Qt::Horizontal);
    gridLayout2->addWidget(sliderScaleAll, 1, 1, 1, 1);

    labelColorAll = new QLabel(groupBoxAll);
    labelColorAll->setObjectName(QString::fromUtf8("labelColorAll"));
    gridLayout2->addWidget(labelColorAll, 2, 0, 1, 1);

    spacerItem1 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    gridLayout2->addItem(spacerItem1, 5, 1, 1, 1);

    labelScaleAll = new QLabel(groupBoxAll);
    labelScaleAll->setObjectName(QString::fromUtf8("labelScaleAll"));
    gridLayout2->addWidget(labelScaleAll, 1, 0, 1, 1);

    checkBoxShowHideAll = new QCheckBox(groupBoxAll);
    checkBoxShowHideAll->setObjectName(QString::fromUtf8("checkBoxShowHideAll"));
    checkBoxShowHideAll->setChecked(true);
    gridLayout2->addWidget(checkBoxShowHideAll, 0, 0, 1, 1);

    sliderColorAll = new QSlider(groupBoxAll);
    sliderColorAll->setObjectName(QString::fromUtf8("sliderColorAll"));
    sliderColorAll->setOrientation(Qt::Horizontal);
    gridLayout2->addWidget(sliderColorAll, 2, 1, 1, 1);

    pushButtonNew = new QPushButton(groupBoxAll);
    pushButtonNew->setObjectName(QString::fromUtf8("pushButtonNew"));
    pushButtonNew->setCheckable(true);
    pushButtonNew->setChecked(false);
    gridLayout2->addWidget(pushButtonNew, 0, 1, 1, 1);

    spacerItem2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    gridLayout2->addItem(spacerItem2, 3, 1, 1, 1);

    gridLayout->addWidget(groupBoxAll, 0, 1, 1, 1);

    listWidget = new QListWidget(w);
    listWidget->setObjectName(QString::fromUtf8("listWidget"));
    gridLayout->addWidget(listWidget, 0, 0, 2, 1);

    //retranslateUi(w);
    groupBoxCurrent->setTitle(QApplication::translate("w", "Current Annotation", 0));
    checkBoxShowHide->setText(QApplication::translate("w", "Show Annotation", 0));
    labelScale->setText(QApplication::translate("w", "Scale Annotation", 0));
    pushButtonDelete->setText(QApplication::translate("w", "Delete", 0));
    labelColor->setText(QApplication::translate("w", "Change Color", 0));
    pushButtonOk->setText(QApplication::translate("w", "OK", 0));
    groupBoxAll->setTitle(QApplication::translate("w", "All Annotations", 0));
    pushButtonDeleteAll->setText(QApplication::translate("w", "Delete All", 0));
    labelColorAll->setText(QApplication::translate("w", "Change Colors", 0));
    labelScaleAll->setText(QApplication::translate("w", "Scale all Annotations", 0));
    checkBoxShowHideAll->setText(QApplication::translate("w", "Show Annotations", 0));
    pushButtonNew->setText(QApplication::translate("w", "New Annotation", 0));

    //setup conncetions to slots
    connect(pushButtonOk, SIGNAL(clicked()), this, SLOT(sendText()));
    connect(pushButtonNew, SIGNAL(clicked()), this, SLOT(newAnnotation()));
    connect(pushButtonDelete, SIGNAL(clicked()), this, SLOT(deleteAnnotation()));
    connect(pushButtonDeleteAll, SIGNAL(clicked()), this, SLOT(deleteAllAnnotations()));

    connect(sliderScale, SIGNAL(sliderMoved(int)), this, SLOT(scaleAnnotation(int)));
    connect(sliderScaleAll, SIGNAL(valueChanged(int)), this, SLOT(scaleAllAnnotations(int)));

    connect(sliderColor, SIGNAL(valueChanged(int)), this, SLOT(setAnnotationColor(int)));
    connect(sliderColorAll, SIGNAL(valueChanged(int)), this, SLOT(setAllAnnotationColors(int)));

    connect(checkBoxShowHide, SIGNAL(stateChanged(int)), this, SLOT(showOrHideAnnotation(int)));
    connect(checkBoxShowHideAll, SIGNAL(stateChanged(int)), this, SLOT(showOrHideAllAnnotations(int)));

    connect(listWidget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(itemClicked(QListWidgetItem *)));

    //QSize size(488, 566);
    //size = size.expandedTo(w->minimumSizeHint());
    //w->resize(size);
}

TUIAnnotationTab::~TUIAnnotationTab()
{
}

//*************************************************************
//*****                 Virtual Funtions                  *****
//*************************************************************
const char *TUIAnnotationTab::getClassName()
{
    return "TUIAnnotationTab";
}

void TUIAnnotationTab::setValue(int type, covise::TokenBuffer &tb)
{
    if (type == TABLET_ANNOTATION_CHANGE_NEW_BUTTON_STATE)
    {
        char state;
        tb >> state;
        pushButtonNew->setChecked(state);
    }

    else if (type == TABLET_ANNOTATION_NEW)
    {
        int newID;
        tb >> newID;

        QString annotationName = "Annotation ";
        std::ostringstream stream;
        if (stream << newID)
        {
            annotationName.append(stream.str().c_str());
        }
        else
        {
            annotationName.append("X");
        }

        newListItem = new QListWidgetItem;
        newListItem->setText(annotationName);

        QVariant variant(newID);
        newListItem->setData(IDRole, variant);

        listWidget->addItem(newListItem);
    }
    else if (type == TABLET_ANNOTATION_DELETE)
    {
        int mode, id;
        tb >> mode;
        tb >> id;
        if (mode)
        {
            QListWidgetItem *item = NULL;

            for (int i = 0; i < listWidget->count(); i++)
            {
                if (listWidget->item(i)->data(IDRole).toInt() == id)
                {
                    item = listWidget->item(i);
                }
            }

            if (item)
                delete item;
        }
        else
        {
            listWidget->clear();
        }
    }
    else if (type == TABLET_ANNOTATION_SET_SELECTION)
    {
        int searchID;
        tb >> searchID;

        bool found = false;
        QListWidgetItem *item = NULL;

        for (int i = 0; i < listWidget->count(); i++)
        {
            if (listWidget->item(i)->data(IDRole).toInt() == searchID)
            {
                item = listWidget->item(i);
                listWidget->setCurrentItem(item);
                found = true;
            }
        }

        if (found)
        {
            textEdit->setText(item->data(TextRole).toString());
        }
    }

    TUIElement::setValue(type, tb);
}

//*************************************************************
//*****                 Private Funtions                  *****
//*************************************************************

//*************************************************************
//*****                 Public Slots                      *****
//*************************************************************
void TUIAnnotationTab::sendText()
{
    QListWidgetItem *curr = listWidget->currentItem();
    if (curr)
    {
        covise::TokenBuffer tb;

        tb << ID;
        tb << TABLET_ANNOTATION_SEND_TEXT;
        QByteArray ba = textEdit->toPlainText().toUtf8();
        tb << ba.data();

        TUIMainWindow::getInstance()->send(tb);

        QVariant variant(ba);

        curr->setData(TextRole, variant);
    }
}

void TUIAnnotationTab::newAnnotation()
{
    //send message including new ID to Plugin
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_NEW;
    if (pushButtonNew->isChecked())
        tb << 1;
    else
        tb << 0;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::deleteAnnotation()
{
    // send delete message to plugin
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_DELETE;

    TUIMainWindow::getInstance()->send(tb);

    //QListWidgetItem *item = listWidget->takeItem(listWidget->currentRow());
    //delete item;
}

void TUIAnnotationTab::deleteAllAnnotations()
{
    // send delete all message to plugin
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_DELETE_ALL;

    TUIMainWindow::getInstance()->send(tb);

    //listWidget->clear();
}

void TUIAnnotationTab::scaleAnnotation(int value)
{
    // send ID and scale message
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SCALE;

    if (listWidget->count() > 0)
        tb << listWidget->currentItem()->data(IDRole).toInt();
    else
        tb << -1;

    tb << ((float)(value + 1)) / 10;
    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::scaleAllAnnotations(int value)
{
    // send scale all message
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SCALE_ALL;
    tb << ((float)(value + 1)) / 10;
    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::setAnnotationColor(int value)
{
    // send ID and setColor message
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SET_COLOR;

    if (listWidget->count() > 0)
        tb << listWidget->currentItem()->data(IDRole).toInt();
    else
        tb << -1;

    tb << (float)value / 100;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::setAllAnnotationColors(int value)
{
    // send setAllColors message
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SET_ALL_COLORS;
    tb << (float)value / 100;
    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::showOrHideAnnotation(int state)
{
    // send ID and hide / show message
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SHOW_OR_HIDE;

    if (listWidget->count() > 0)
        tb << listWidget->currentItem()->data(IDRole).toInt();
    else
        tb << -1;

    tb << state;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::showOrHideAllAnnotations(int state)
{
    // send show / hide all message
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SHOW_OR_HIDE_ALL;
    tb << state;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIAnnotationTab::itemClicked(QListWidgetItem *item)
{
    //inform annotation plugin about selection
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_ANNOTATION_SET_SELECTION;
    tb << item->data(IDRole).toInt();

    TUIMainWindow::getInstance()->send(tb);

    textEdit->setText(item->data(TextRole).toString());
}
