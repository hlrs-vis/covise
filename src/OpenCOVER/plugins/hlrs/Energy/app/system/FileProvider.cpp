#include "FileProvider.h"

#include <filesystem>
#include <algorithm>

std::string FileProvider::createFilePath(int scenarioId, EnergyType type, const std::string &filename)
{
    std::string folder = (scenarioId == -1) ? "static" : std::to_string(scenarioId);
    std::string typeFolder(EnergyTypeToString(type));
    std::string filePath = m_dirPath + "/" + folder + "/" + typeFolder;

    if (filename != "")
        filePath += "/" + filename;
    return filePath;
}

std::vector<std::string> FileProvider::createFilePaths(int scenarioID, EnergyType type, const std::string &extension)
{
    auto filePath = createFilePath(scenarioID, type);
    auto files = discoverFiles(filePath, extension);
    std::for_each(files.begin(), files.end(), [&](auto &file){ createFilePath(scenarioID, type, file); });
    return files;
}

std::vector<std::string> FileProvider::discoverFiles(const std::string &dirPath, const std::string &extension)
{
    std::vector<std::string> files(0);
    if (std::filesystem::exists(dirPath))
        for (const auto &entry : std::filesystem::directory_iterator(dirPath))
            if (entry.is_regular_file() && entry.path().extension() == extension)
                files.emplace_back(entry.path().filename().string());
    return files;
}
