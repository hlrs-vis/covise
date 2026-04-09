#include "CSVLoader.h"
#include <utils/read/csv/csv.h>
#include <filesystem>

namespace
{
std::unique_ptr<opencover::utils::read::CSVStream> createStream(const std::string &path)
{
    auto csvStream = std::make_unique<opencover::utils::read::CSVStream>(path);
    if (!csvStream->is_open())
        throw std::runtime_error("Could not open file: " + path);

    return std::move(csvStream);
}
}

DataPackages CSVLoader::load(int scenarioId, EnergyType type)
{
    CSVDataMap packages;
    auto files = createFilePaths(scenarioId, type, ".csv");
    for (const auto &filePath : files) {
        auto fileName = std::filesystem::path(filePath);
        packages.emplace(fileName.stem(), std::move(createStream(filePath)));
    }
    return packages;
}
