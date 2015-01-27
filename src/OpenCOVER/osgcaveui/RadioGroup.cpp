/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RadioGroup.h"

using namespace cui;

RadioGroup::RadioGroup()
    : CardListener()
{
    _active = 0;
}

RadioGroup::~RadioGroup()
{
    _buttons.clear();
}

void RadioGroup::add(RadioButton *button)
{
    _buttons.push_back(button);
    button->addCardListener(this);
    button->setChecked(false);
}

void RadioGroup::remove(RadioButton *button)
{
    std::vector<RadioButton *>::iterator iter;

    for (iter = _buttons.begin(); iter != _buttons.end(); iter++)
    {
        if ((*iter) == button)
        {
            _buttons.erase(iter);
            break;
        }
    }
}

void RadioGroup::setSelected(RadioButton *button)
{
    std::list<RadioGroupListener *>::iterator iter2;

    if (_active == 0)
    {
        _active = button;
        _active->setChecked(true);

        //       for (iter2 = _listeners.begin(); iter2 != _listeners.end(); iter2++)
        // 	(*iter2)->radioGroupStatusChanged();
    }
}

RadioButton *RadioGroup::getSelected()
{
    return _active;
}

bool RadioGroup::isSelected(RadioButton *button)
{
    return (_active == button);
}

void RadioGroup::addRadioGroupListener(RadioGroupListener *listener)
{
    _listeners.push_back(listener);
}

bool RadioGroup::cardButtonEvent(Card *card, int, int)
{
    std::vector<RadioButton *>::iterator iter1;
    std::list<RadioGroupListener *>::iterator iter2;

    for (iter1 = _buttons.begin(); iter1 != _buttons.end(); iter1++)
    {
        if (*iter1 == card)
        {
            if (_active == (*iter1))
                return false;

            if (_active != 0)
                _active->setChecked(false);
            _active = dynamic_cast<RadioButton *>(card);
            _active->setChecked(true);

            for (iter2 = _listeners.begin(); iter2 != _listeners.end(); iter2++)
                (*iter2)->radioGroupStatusChanged(this);

            return true;
        }
    }

    return false;
}

bool RadioGroup::cardCursorUpdate(Card *, InputDevice *)
{
    return false;
}

void RadioGroup::setVisible(bool flag)
{
    std::vector<RadioButton *>::iterator iter;

    for (iter = _buttons.begin(); iter != _buttons.end(); iter++)
        (*iter)->setVisible(flag);
}
