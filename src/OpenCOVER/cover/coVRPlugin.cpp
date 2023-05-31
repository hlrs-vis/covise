/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPlugin.h"
#include "coVRPluginList.h"
#include <cassert>
#include <iostream>
#include <OpenConfig/access.h>

// #ifndef COVER_PLUGIN_NAME
// #define COVER_PLUGIN_NAME ""
// #endif

using namespace opencover;

coVRPlugin::coVRPlugin(const std::string &name)
:m_name(name)
{
    assert(!m_name.empty());
}

void coVRPlugin::commitTimestep(int t)
{
    if (m_outstandingTimestep != t)
    {
        std::cerr << "Plugin " << m_name << ": timestep " << t << " committed, but " << m_outstandingTimestep << " was outstanding" << std::endl;
    }
    assert(m_outstandingTimestep == -1 || m_outstandingTimestep == t);
    m_outstandingTimestep = -1;
    coVRPluginList::instance()->commitTimestep(t, this);
}

void coVRPlugin::requestTimestepWrapper(int t)
{
    assert(m_outstandingTimestep == -1);
    m_outstandingTimestep = t;

    requestTimestep(t);
}

std::shared_ptr<config::File> coVRPlugin::config()
{
    if (!m_configFile)
    {
        config::Access acc;
        m_configFile = acc.file(std::string("plugin/") + getName());
    }
    return m_configFile;
}

template<class V>
std::unique_ptr<config::Value<V>> coVRPlugin::config(const std::string &section, const std::string &name,
                                                     const V &defVal, config::Flag flags)
{
    return config()->value<V>(section, name, defVal, flags);
}

std::unique_ptr<config::Value<bool>> coVRPlugin::configBool(const std::string &section, const std::string &name,
                                                            const bool &defVal, config::Flag flags)
{
    return config(section, name, defVal, flags);
}

std::unique_ptr<config::Value<int64_t>> coVRPlugin::configInt(const std::string &section, const std::string &name,
                                                              const int64_t &defVal, config::Flag flags)
{
    return config(section, name, defVal, flags);
}

std::unique_ptr<config::Value<double>> coVRPlugin::configFloat(const std::string &section, const std::string &name,
                                                               const double &defVal, config::Flag flags)
{
    return config(section, name, defVal, flags);
}

std::unique_ptr<config::Value<std::string>> coVRPlugin::configString(const std::string &section,
                                                                     const std::string &name, const std::string &defVal,
                                                                     config::Flag flags)
{
    return config(section, name, defVal, flags);
}

template<class V>
std::unique_ptr<config::Array<V>> coVRPlugin::configArray(const std::string &section, const std::string &name,
                                                          const std::vector<V> &defVal, config::Flag flags)
{
    return config()->array<V>(section, name, defVal, flags);
}

std::unique_ptr<config::Array<bool>> coVRPlugin::configBoolArray(const std::string &section, const std::string &name,
                                                                 const std::vector<bool> &defVal, config::Flag flags)
{
    return configArray(section, name, defVal, flags);
}

std::unique_ptr<config::Array<int64_t>> coVRPlugin::configIntArray(const std::string &section, const std::string &name,
                                                                   const std::vector<int64_t> &defVal,
                                                                   config::Flag flags)
{
    return configArray(section, name, defVal, flags);
}

std::unique_ptr<config::Array<double>> coVRPlugin::configFloatArray(const std::string &section, const std::string &name,
                                                                    const std::vector<double> &defVal,
                                                                    config::Flag flags)
{
    return configArray(section, name, defVal, flags);
}

std::unique_ptr<config::Array<std::string>> coVRPlugin::configStringArray(const std::string &section,
                                                                          const std::string &name,
                                                                          const std::vector<std::string> &defVal,
                                                                          config::Flag flags)
{
    return configArray(section, name, defVal, flags);
}
