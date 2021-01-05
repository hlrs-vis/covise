/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "roadtypecombobox.h"

// Qt //
//
#include <QGridLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QLabel>

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

//################//
//                //
// Road Type Combo Box //
//                //
//################//

void RoadTypeComboBox::initComboBox()
{
    // List //
    //
    QStringList roadTypeNames;
    roadTypeNames << QLabel::tr("unknown") << QLabel::tr("rural") << QLabel::tr("motorway") << QLabel::tr("town") << QLabel::tr("lowspeed") << QLabel::tr("pedestrian");

    // Settings //
    //

    comboBox_ = new QComboBox;
    comboBox_->addItems(roadTypeNames);
    comboBox_->setCurrentIndex(0);
    QList<QColor> colors;
    colors
        << ODD::instance()->colors()->brightGrey()
        << ODD::instance()->colors()->brightOrange()
        << ODD::instance()->colors()->brightRed()
        << ODD::instance()->colors()->brightGreen()
        << ODD::instance()->colors()->brightCyan()
        << ODD::instance()->colors()->brightBlue();

    for (int i = 0; i < roadTypeNames.count(); ++i)
    {
        QPixmap icon(12, 12);
        icon.fill(colors.at(i));
        comboBox_->setItemIcon(i, icon);
    }

    gridLayout_ = new QGridLayout;
    QLabel *label = new QLabel(QLabel::tr("Road Type"));
    gridLayout_->addWidget(label, 0, 0);
    gridLayout_->addWidget(comboBox_, 1, 0);
}
