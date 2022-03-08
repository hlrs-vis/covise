#include "coConfigTools.h"
#include <config/coConfigConstants.h>
#include <config/coConfigLog.h>
#include <util/string_util.h>
using namespace covise;

const std::array<const char*, 4> coConfigTools::attributeNames = {"MASTER", "HOST", "ARCH", "RANK"}; // order is important

bool coConfigTools::matchingAttributes(const std::map<std::string, std::string> &attributes)
{
    if (attributes.empty())
        return true;

    std::array<bool (*)(const std::string &), 4> matchFunctions = {&matchingMaster, &matchingHost, &matchingArch, &matchingRank};
    for (size_t i = 0; i < attributeNames.size(); i++)
    {
        auto it = attributes.find(attributeNames[i]);
        if (it != attributes.end() && !matchFunctions[i](it->second))
            return false;
    }
    return true;
}

bool coConfigTools::matchingHost(const std::string &host)
{
    if (coConfigConstants::getHostname().empty())
        return true;

    auto hosts = split(host, ',', true);
    for (const auto &ho : hosts)
    {
        std::string h = toLower(strip(ho));
        if (h == coConfigConstants::getHostname())
            return true;
        if (h.substr(0, h.find('.')) == coConfigConstants::getHostname())
            return true;
        if (h == coConfigConstants::getHostname().substr(0, coConfigConstants::getHostname().find('.')))
            return true;
    }

    COCONFIGDBG("coConfigEntry::matchingHost info: host " << host << " not matching " << coConfigConstants::getHostname());
    return false;
}

bool coConfigTools::matchingMaster(const std::string &master)
{
    if (coConfigConstants::getMaster().empty())
        return true;

    auto masters = split(master, ',', true);
    for (const auto &it : masters)
    {
        std::string m = toLower(strip(it));
        if (m == coConfigConstants::getMaster())
            return true;
        if (m.substr(0, m.find('.')) == coConfigConstants::getMaster())
            return true;
        if (m == coConfigConstants::getMaster().substr(0, coConfigConstants::getMaster().find('.')))
            return true;
    }

    COCONFIGDBG("coConfigEntry::matchingMaster info: master " << master << " not matching " << coConfigConstants::getMaster());
    return false;
}

bool coConfigTools::matchingArch(const std::string &arch)
{
    if (coConfigConstants::getArchList().find(arch) == coConfigConstants::getArchList().end())
    {
        COCONFIGDBG("coConfigEntry::matchingArch info: arch " << arch << " not matching");
        return false;
    }
    return true;
}

bool coConfigTools::matchingRank(const std::string &rank)
{
    bool match = false;
    std::string r = strip(rank);
    auto ranges = split(r, ',', true);
    for (auto &range : ranges)
    {
        range = strip(range);
        if (range == "-1" || toLower(range) == "all" || toLower(range) == "any")
        {
            match = true;
        }
        else if (range.find("-") != std::string::npos)
        {
            auto ext = split(range, '-', true);
            if (ext.size() != 2)
            {
                COCONFIGDBG("coConfigEntry::matchingRank info: cannot parse range of ranks " << range);
            }
            else if (coConfigConstants::getRank() < atoi(ext[0].c_str()) || coConfigConstants::getRank() > atoi(ext[1].c_str()))
            {
                COCONFIGDBG("coConfigEntry::matchingRank info: range of ranks " << range << " not matching");
            }
            else
            {
                match = true;
            }
        }
        else
        {
            if (coConfigConstants::getRank() == atoi(range.c_str()))
                match = true;
        }
    }

    if (!match)
    {
        COCONFIGDBG("coConfigEntry::matchingRank info: rank " << rank << " not matching");
    }

    return match;
}
