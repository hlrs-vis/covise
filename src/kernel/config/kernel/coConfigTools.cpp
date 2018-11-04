#include "coConfigTools.h"
#include <config/coConfigConstants.h>
#include <config/coConfigLog.h>

using namespace covise;

bool coConfigTools::matchingAttributes(QHash<QString, QString *> attributes)
{
    const QString *master = attributes["MASTER"];
    if (!matchingMaster(master))
        return false;

    const QString *host = attributes["HOST"];
    if (!matchingHost(host))
        return false;

    const QString *arch = attributes["ARCH"];
    if (!matchingArch(arch))
        return false;

    const QString *rank = attributes["RANK"];
    if (!matchingRank(rank))
        return false;

    return true;
}

bool coConfigTools::matchingHost(const QString *host)
{
    if (!host)
        return true;

    if (coConfigConstants::getHostname().isEmpty())
        return true;

    QStringList hosts = host->split(',', QString::SkipEmptyParts);
    for (QStringList::iterator it = hosts.begin(); it != hosts.end(); ++it)
    {
        QString h = it->trimmed().toLower();
        if (h == coConfigConstants::getHostname())
            return true;
        if (h.section('.', 0, 0) == coConfigConstants::getHostname())
            return true;
        if (h == coConfigConstants::getHostname().section('.', 0, 0))
            return true;
    }

    COCONFIGDBG("coConfigEntry::matchingHost info: host " << *host << " not matching " << coConfigConstants::getHostname());
    return false;
}

bool coConfigTools::matchingMaster(const QString *master)
{
    if (!master)
        return true;

    if (coConfigConstants::getMaster().isEmpty())
        return true;

    QStringList masters = master->split(',', QString::SkipEmptyParts);
    for (QStringList::iterator it = masters.begin(); it != masters.end(); ++it)
    {
        QString m = it->trimmed().toLower();
        if (m == coConfigConstants::getMaster())
            return true;
        if (m.section('.', 0, 0) == coConfigConstants::getMaster())
            return true;
        if (m == coConfigConstants::getMaster().section('.', 0, 0))
            return true;
    }

    COCONFIGDBG("coConfigEntry::matchingMaster info: master " << *master << " not matching " << coConfigConstants::getMaster());
    return false;
}

bool coConfigTools::matchingArch(const QString *arch)
{
    if (!arch)
        return true;

    if (!coConfigConstants::getArchList().contains(*arch))
    {
        COCONFIGDBG("coConfigEntry::matchingArch info: arch " << *arch << " not matching");
        return false;
    }

    return true;
}

bool coConfigTools::matchingRank(const QString *rank)
{
    if (!rank)
        return true;

    bool match = false;
    QString r = rank->simplified();
    QStringList ranges = r.split(",");
    for (int i =0; i<int(ranges.size()); ++i)
    {
        QString range = ranges[i].simplified();
        if (range == "-1" || range.toLower() == "all" || range.toLower() == "any")
        {
            match = true;
        }
        else if (range.contains("-"))
        {
            QStringList ext = range.split("-");
            if (ext.size() != 2)
            {
                COCONFIGDBG("coConfigEntry::matchingRank info: cannot parse range of ranks " << range);
            }
            else if (coConfigConstants::getRank() < ext[0].toInt() || coConfigConstants::getRank() > ext[1].toInt())
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
            if (coConfigConstants::getRank() == range.toInt())
                match = true;
        }
    }

    if (!match)
    {
        COCONFIGDBG("coConfigEntry::matchingRank info: rank " << *rank << " not matching");
    }

    return match;
}
