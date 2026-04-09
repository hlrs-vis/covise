#include "ArrowLoader.h"
#include <filesystem>
#include <utils/read/apache/arrow.h>

DataPackages ArrowLoader::load(int scenarioId, EnergyType type)
{
    ArrowDataMap packages;
    auto files = createFilePaths(scenarioId, type, ".arrow");
    for (const auto &filePath : files)
    {
        auto fileName = std::filesystem::path(filePath);
        apache::ArrowReader reader(filePath);
        packages.emplace(fileName.stem(), std::move(reader.getTable()));
    }
    return packages;
}
