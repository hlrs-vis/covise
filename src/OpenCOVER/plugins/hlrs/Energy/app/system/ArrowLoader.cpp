#include "ArrowLoader.h"
#include <filesystem>
#include <utils/read/apache/arrow.h>

DataPackages ArrowLoader::load(const Scenario &scenario, EnergyType type) const
{
    ArrowDataMap packages;
    auto files = createFilePaths(scenario, type, ".arrow");
    for (const auto &filePath : files)
    {
        auto fileName = std::filesystem::path(filePath);
        apache::ArrowReader reader(filePath);
        packages.emplace(fileName.stem(), std::move(reader.getTable()));
    }
    return packages;
}
