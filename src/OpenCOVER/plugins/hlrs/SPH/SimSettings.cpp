/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <map>
#include <iostream>

using namespace std;

class SimSettings
{
public:
public
    SimSettings()
    {
    }

protected:
private:
    multimap<char, int> mymm;
    multimap<char, int>::iterator it;

    typedef multimap<string, string>::type SettingsMultiMap;
    SettingsMultiMap Settings;
};