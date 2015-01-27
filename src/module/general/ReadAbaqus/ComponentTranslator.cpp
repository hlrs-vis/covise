/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "ComponentTranslator.h"
#include "Data.h"

ComponentTranslator::ComponentTranslator(const odb_SequenceString &secStr)
{
    switch (Data::TYPE)
    {
    case Data::VECTOR:
        _size = 3;
        _translator = new int[3];
        break;
    case Data::TENSOR:
        _size = 6;
        _translator = new int[6];
        break;
    default:
        _size = 0;
        _translator = NULL;
        return;
    }
    if (_size > 0)
    {
        int i;
        for (i = 0; i < _size; ++i)
        {
            _translator[i] = -1;
        }
        for (i = 0; i < secStr.size(); ++i)
        {
            if (Data::TYPE == Data::VECTOR)
            {
                const char *label = secStr.constGet(i).CStr();
                int label_len = strlen(label);
                int num = -1;
                sscanf(label + label_len - 1, "%d", &num);
                _translator[i] = num - 1;
            }
            else if (Data::TYPE == Data::TENSOR)
            {
                const char *label = secStr.constGet(i).CStr();
                int label_len = strlen(label);
                int num = -1;
                sscanf(label + label_len - 2, "%d", &num);
                switch (num)
                {
                case 11:
                    _translator[i] = 0;
                    break;
                case 22:
                    _translator[i] = 1;
                    break;
                case 33:
                    _translator[i] = 2;
                    break;
                case 12:
                case 21:
                    _translator[i] = 3;
                    break;
                case 23:
                case 32:
                    _translator[i] = 4;
                    break;
                case 13:
                case 31:
                    _translator[i] = 5;
                    break;
                }
            }
        }
    }
}

ComponentTranslator::~ComponentTranslator()
{
    delete[] _translator;
}

int
    ComponentTranslator::
    operator[](int i) const
{
    if (i >= 0 && i < _size)
    {
        return _translator[i];
    }
    return -1;
}
