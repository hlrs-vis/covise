/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "src/util/roadtypecombobox.h"

#include <QWidget>
#include <QLabel>
#include <QDoubleSpinBox>

class TypeSectionSettingsUI
{
public:
    TypeSectionSettingsUI();
    ~TypeSectionSettingsUI(){};

    void setupUI(QWidget *TypeSectionSettings);
    RoadTypeComboBox *getRoadTypeComboBox()
    {
        return roadTypeComboBox_;
    };
    QLabel *getLabel()
    {
        return maxSpeedLabel;
    };
    QDoubleSpinBox *getSpinBox()
    {
        return spinBox;
    };

private:
    RoadTypeComboBox *roadTypeComboBox_;
    QLabel *maxSpeedLabel;
    QDoubleSpinBox *spinBox;
};