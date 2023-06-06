#include "SidecarConfigBridge.h"
#include <OpenConfig/array.h>
#include <OpenConfig/value.h>
#include <locale>
#include <iostream>

using namespace opencover;

SidecarConfigBridge::Key::Key() = default;
SidecarConfigBridge::Key::Key(const opencover::config::ConfigBase *entry)
: path(entry->path()), section(entry->section()), name(entry->name())
{
    //std::cerr << "Key: " << configSection() << ":" << configName() << std::endl;
}

bool SidecarConfigBridge::Key::operator<(const SidecarConfigBridge::Key &o) const
{
    if (path == o.path)
    {
        if (section == o.section)
        {
            return name < o.name;
        }
        return section < o.section;
    }
    return path < o.path;
}

std::string SidecarConfigBridge::Key::configSection() const
{
    std::locale C("C");
    std::string cleanPath;
    for (auto c: path)
    {
        if (std::isalnum(c, C))
        {
            cleanPath.push_back(c);
        }
        else if (c == '/')
        {
            cleanPath.push_back('-');
        }
    }
    std::string sect = cleanPath + ":" + section;
    return sect;
}

std::string SidecarConfigBridge::Key::configName() const
{
    return name;
}


std::ostream &operator<<(std::ostream &os, const SidecarConfigBridge::Key &key)
{
    os << key.path << ":" << key.section << ":" << key.name;
    return os;
}

namespace
{
template<class V>
std::ostream &operator<<(std::ostream &os, const std::vector<V> &vec)
{
    os << "{";
    for (const auto &val: vec)
    {
        os << " " << val;
    }
    os << " }";
    return os;
}
} // namespace


SidecarConfigBridge::SidecarConfigBridge(const std::string &file, bool saveOnDestroy)
: m_file(file + ".cover"), m_save(saveOnDestroy)
{
    try
    {
        m_toml = toml::parse_file(m_file);
    }
    catch (std::exception &ex)
    {
        std::cerr << "could not load sidecar file " << m_file << ": " << ex.what() << std::endl;
    }
}

static void pruneEmptySections(toml::v3::table *tbl)
{
    if (!tbl)
        return;
    auto it = tbl->begin();
    while (it != tbl->end())
    {
        toml::v3::table *t = it->second.as_table();
        if (t)
        {
            pruneEmptySections(t);
            if (t->empty())
            {
                it = tbl->erase(it);
            }
            else
            {
                ++it;
            }
        }
        else
        {
            ++it;
        }
    }
}

SidecarConfigBridge::~SidecarConfigBridge()
{
    if (!m_save)
        return;
    pruneEmptySections(&m_toml);
    std::string temp = m_file + ".new";
    try
    {
        if (!m_toml.empty())
        {
            std::ofstream f(temp);
            f << m_toml;
            f.close();
        }
        std::string backup = m_file + ".backup";
        std::remove(backup.c_str());
        std::rename(m_file.c_str(), backup.c_str());
        std::remove(m_file.c_str());
        if (!m_toml.empty() && std::rename(temp.c_str(), m_file.c_str()) != 0)
        {
            std::cerr << "failed to move updated config to " << m_file << std::endl;
        }
    }
    catch (std::exception &ex)
    {
        std::cerr << "failed to save config to " << temp << ": " << ex.what() << std::endl;
    }
}

// store values to TOML, unless when seen for the first time: then retrieve from TOML and set value, for both Value<V>'s and Array<V>'s
template<class V>
bool SidecarConfigBridge::toOrFromToml(const opencover::config::ConfigBase *entry, const SidecarConfigBridge::Key &key,
                                       bool seen)
{
    auto tbl = m_toml[key.configSection()].as_table();
    if (!tbl)
    {
        m_toml.insert(key.configSection(), toml::table());
        tbl = m_toml[key.configSection()].as_table();
        if (!tbl)
            return false;
    }
    toml::node *node = nullptr;
    if (!seen)
    {
        auto it = tbl->find(key.configName());
        if (it != tbl->end())
        {
            node = &it->second;
        }
    }
    if (const auto *value = dynamic_cast<const opencover::config::Value<V> *>(entry))
    {
        if (!seen && node)
        {
            if (auto v = node->as<V>())
            {
                // create a writable copy
                config::Access acc;
                auto val = acc.value<V>(entry->path(), entry->section(), entry->name());
                V vv = v->get();
                *val = vv;
                return true;
            }
            tbl->erase(key.configName());
        }
        m_modified = true;
        if (value->value() == value->defaultValue())
        {
            tbl->erase(key.configName());
            return true;
        }
        tbl->insert_or_assign(key.configName(), value->value());
        return true;
    }

    if (const auto *array = dynamic_cast<const opencover::config::Array<V> *>(entry))
    {
        if (!seen && node)
        {
            if (auto a = node->as_array())
            {
                // create a writable copy
                config::Access acc;
                auto arr = acc.array<V>(entry->path(), entry->section(), entry->name());
                bool ok = true;
                std::vector<V> vals;
                for (auto &v: *a)
                {
                    if (auto vv = v.as<V>())
                    {
                        vals.push_back(vv->get());
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                {
                    *arr = vals;
                    return true;
                }
                tbl->erase(key.configName());
            }
        }
        m_modified = true;
        if (array->value() == array->defaultValue())
        {
            tbl->erase(key.configName());
            return true;
        }
        toml::array arr;
        auto val = array->value();
        for (auto v: val)
            arr.push_back(V(v));
        tbl->insert_or_assign(key.configName(), arr);
        return true;
    }

    return false;
}

bool SidecarConfigBridge::wasChanged(const opencover::config::ConfigBase *entry)
{
    auto path = entry->path();
    auto section = entry->section();
    auto name = entry->name();
    Key key(entry);

    bool seen = true;
    auto it = m_seen.find(key);
    if (it == m_seen.end())
    {
        it = m_seen.emplace(key).first;
        seen = false;
    }

    toOrFromToml<bool>(entry, key, seen);
    toOrFromToml<int64_t>(entry, key, seen);
    toOrFromToml<double>(entry, key, seen);
    toOrFromToml<std::string>(entry, key, seen);

    return true;
}
