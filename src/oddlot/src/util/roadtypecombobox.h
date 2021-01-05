/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ROADTYPECOMBOBOX_HPP
#define ROADTYPECOMBOBOX_HPP

#include <QGridLayout>
#include <QComboBox>

class RoadTypeComboBox
{
public:
    RoadTypeComboBox()
    {
        initComboBox();
    };
    ~RoadTypeComboBox(){};

    QGridLayout *getGridLayout()
    {
        return gridLayout_;
    };
    QComboBox *getComboBox()
    {
        return comboBox_;
    };

private:
    QComboBox *comboBox_;
    QGridLayout *gridLayout_;

    void initComboBox();
};

#endif // ROADTYPECOMBOBOX_HPP