/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "typesectionsettingsUI.h"

#include <QMetaObject>
#include <QGridLayout>

TypeSectionSettingsUI::TypeSectionSettingsUI()
{
}

void TypeSectionSettingsUI::setupUI(QWidget *TypeSectionSettings)
{
    QGridLayout *gridLayout;
    QComboBox *roadTypeBox;

    if (TypeSectionSettings->objectName().isEmpty())
        TypeSectionSettings->setObjectName(QString::fromUtf8("TypeSectionSettings"));
    TypeSectionSettings->resize(807, 544);
    gridLayout = new QGridLayout(TypeSectionSettings);
    gridLayout->setObjectName(QString::fromUtf8("gridLayout"));

    roadTypeComboBox_ = new RoadTypeComboBox();
    roadTypeBox = roadTypeComboBox_->getComboBox();
    roadTypeBox->setObjectName(QString::fromUtf8("roadTypeBox"));

    gridLayout->addWidget(roadTypeBox);

    //	selectGroupBox_ = new QGroupBox(tr("Add Settings"));
    //	selectGroupBox_->setLayout(roadTypeBox_->getGridLayout());
    //	selectGroupBox_->setEnabled(false);

    //	toolLayout->addWidget(selectGroupBox_, ++row, 0);

    QMetaObject::connectSlotsByName(TypeSectionSettings);
}
