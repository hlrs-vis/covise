/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_HOST_TYPE_H
#define _CO_HOST_TYPE_H

// type of host, CO_HOST/CO_PARTNER result of add partner or add host
//               CO_*_MIRROR        result during mirrored session
typedef enum
{
    CO_HOST,
    CO_PARTNER,
    CO_HOST_MIRROR,
    CO_PARTNER_MIRROR
} co_Host_Type;

class HType
{

public:
    HType(co_Host_Type type)
    {
        _type = type;
        if (_type == CO_HOST)
        {
            _type_msg[0] = 'H';
            _type_msg[1] = 0;
        }
        else
        {
            _type_msg[0] = 'P';
            _type_msg[1] = 0;
        }
    }
    HType(const char *msg)
    {
        if (msg[0] == 'H')
        {
            _type = CO_HOST;
        }
        else
        {
            _type = CO_PARTNER;
        }
    }
    HType(const HType &h)
    {
        _type = h._type;
        _type_msg[0] = h._type_msg[0];
        _type_msg[1] = 0;
    }
    HType &operator=(const HType &h)
    {
        _type = h._type;
        _type_msg[0] = h._type_msg[0];
        _type_msg[1] = 0;
        return *this;
    }
    const char *get_msg() const
    {
        return _type_msg;
    }
    co_Host_Type get_type() const
    {
        return _type;
    }
    void mirror()
    {
        switch (_type)
        {
        case CO_HOST:
        {
            _type = CO_HOST_MIRROR;
            break;
        }
        case CO_PARTNER:
        {
            _type = CO_PARTNER_MIRROR;
            break;
        }
        case CO_HOST_MIRROR:
        {
            _type = CO_HOST;
            break;
        }
        case CO_PARTNER_MIRROR:
        {
            _type = CO_PARTNER;
            break;
        }
        }
    }

protected:
private:
    co_Host_Type _type;
    char _type_msg[2];
};
typedef HType coHostType;
#endif
