/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "src/util/roadtypecombobox.h"

#include <QWidget>

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

private:
    RoadTypeComboBox *roadTypeComboBox_;
};