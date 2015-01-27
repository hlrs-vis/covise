/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __HLERROR_H__
#define __HLERROR_H__

#include <string>
using std::string;

class HlError
{

private:
    string mDescription;
    int mState;

public:
    HlError();
    HlError(HlError &err);
    HlError(int state, const string &desc);
    void setDescription(const string &desc);
    const string &getDescription();
    void setState(int state);
    void setState(HlError &err);
    int getState();
    bool noError();
    void setNoError();
};

inline void HlError::setState(int state)
{
    mState = state;
}

inline int HlError::getState()
{
    return mState;
}

inline const string &HlError::getDescription()
{
    return mDescription;
}

inline void HlError::setDescription(const string &desc)
{
    mDescription = desc;
}

inline void HlError::setState(HlError &err)
{
    setState(err.getState());
    setDescription(err.getDescription());
}

inline void HlError::setNoError()
{
    setState(0);
    setDescription("OK");
}

inline HlError::HlError()
{
    setNoError();
}

inline HlError::HlError(HlError &err)
{
    setState(err);
}

inline HlError::HlError(int state, const string &desc)
{
    setState(state);
    setDescription(desc);
}

inline bool HlError::noError()
{
    return getState() == 0;
}

#endif // __HLERROR__
