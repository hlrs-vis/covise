/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SimSettings_h__
#define __SimSettings_h__

#include <map>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
//#include <algorithm> // sort, max_element, random_shuffle, remove_if, lower_bound
//#include <functional> // greater, bind2nd

using namespace std;

namespace SimLib
{

template <typename T>
struct Setting
{
    T min;
    T max;
    T value;
    std::string unit;
    bool editable;
};
//
class SettingsChangeCallback
{
public:
    virtual void SettingChanged(std::string settingName) = 0;
};

class SimSettings
{
private:
    typedef std::map<std::string, Setting<float> > SettingsType;
    SettingsType mSettingsMap;

    typedef std::vector<SettingsChangeCallback *> CallbacksType;
    CallbacksType mCallbacks;

public:
    SimSettings()
    {
    }
    ~SimSettings()
    {
    }

    void AddCallback(SettingsChangeCallback *callbackClass)
    {
        mCallbacks.insert(mCallbacks.begin(), callbackClass);
    }

    void RemoveCallback(SettingsChangeCallback *callbackClass)
    {
        //mCallbacks.erase(std::remove(mCallbacks.begin(), mCallbacks.end(), callbackClass), mCallbacks.end());
    }

    void AddSetting(std::string settingName, float value, float minValue, float maxValue, std::string unitType, bool editable)
    {
        Setting<float> s;
        s.value = value;
        s.min = minValue;
        s.max = maxValue;
        s.unit = unitType;
        s.editable = editable;
        AddSetting(settingName, s);
    }
    void AddSetting(std::string settingName, float value, float minValue, float maxValue, std::string unitType)
    {
        Setting<float> s;
        s.value = value;
        s.min = minValue;
        s.max = maxValue;
        s.unit = unitType;

        AddSetting(settingName, s);
    }

    void AddSetting(std::string settingName, Setting<float> s)
    {
        mSettingsMap[settingName] = s;
    }
    void SetMin(std::string settingName, float value)
    {
        mSettingsMap[settingName].min = value;
    }

    void SetMax(std::string settingName, float value)
    {
        mSettingsMap[settingName].max = value;
    }

    void SetValue(std::string settingName, float value)
    {
        if (mSettingsMap.find(settingName) == mSettingsMap.end())
        {
            cout << "SimSettings::SetValue (" << settingName << ") failed, no such setting\n";
            return;
        }

        float oldVal = mSettingsMap[settingName].value;

        if (oldVal != value)
        {
            cout << "SETTING: " << setw(30) << settingName << " " << setw(15) << oldVal << setw(5) << " -> " << setw(15) << value << "\n";
            mSettingsMap[settingName].value = value;
            for (CallbacksType::const_iterator it = mCallbacks.begin(); it != mCallbacks.end(); ++it)
            {
                (*it)->SettingChanged(settingName);
            }
        }
    }

    float GetValue(std::string settingName)
    {
        if (mSettingsMap.find(settingName) == mSettingsMap.end())
        {
            cout << "SimSettings::GetValue(" + settingName + ") failed, no such setting\n";
            return 0.0;
        }
        return mSettingsMap[settingName].value;
    }

    void Print()
    {
        cout << "\n*** Simulation Library Settings ***:\n";
        for (SettingsType::const_iterator it = mSettingsMap.begin(); it != mSettingsMap.end(); ++it)
        {
            cout << setw(30) << it->first;
            cout << setw(5) << "\t";
            cout << it->second.value;
            cout << "\n";
        }
        cout << "\n";
    }
};

} // namespace SimLib

#endif
