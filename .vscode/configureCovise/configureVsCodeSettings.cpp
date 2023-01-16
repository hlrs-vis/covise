
#include <iostream>
#include <fstream>
#include <string>
#include <exception>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Using user-defined (raw) string literals
using namespace nlohmann::literals;

int main(int argc, char**argv)
{
    std::string covisedir = argv[1];
    std::string archsuffix = argv[2];
    std::string externlibs = argv[3];
    std::string qtVersion = argv[4];
    std::string generator = argv[5];
    bool overwrite = argv[6] == std::string("overwrite");

    //for some option the ${config:...} evaluation fails so that they have to be set here 
    json coviseEnv = {{"COVISEDIR", "${workspaceFolder}"},
                      {"EXTERNLIBS", argv[3]},
                      {"COVISE_PATH", "${workspaceFolder}"},
                      {"COVISEDESTDIR", "${workspaceFolder}"},
                      {"ARCHSUFFIX", archsuffix}};


    json dynamicCoviseSettings = {
        {"covise.dependencyPath", externlibs},
        {"covise.qtVersion", qtVersion},
        {"cmake.generator", generator},
        {"covise.archsuffix", archsuffix},
#ifdef _WIN32
        {"cmake.cmakePath", "${workspaceFolder}/.vscode/configureCovise/CMakeWrapper.bat"},
#else
        {"cmake.environment", coviseEnv},
        {"cmake.buildDirectory", "${workspaceFolder}/" + archsuffix + "/build.covise"}
#endif

    };
    covisedir += "/.vscode/";
    std::string userSettingsFileName = covisedir + "settings.json";
    std::string coviseSettingsFileName = covisedir + "configureCovise/settings.json";
    std::string userLaunchFileName = covisedir + "launch.json";
    std::string coviseLaunchFileName = covisedir + "configureCovise/launch.json";

    std::ifstream userSettingsFile(userSettingsFileName);
    std::ifstream coviseSettingsFile(coviseSettingsFileName);
    std::ifstream userLaunchFile(userLaunchFileName);
    std::ifstream coviseLaunchFile(coviseLaunchFileName);

    if(!coviseSettingsFile.is_open() || !coviseLaunchFile.is_open())
    {
        std::cerr << "failed to open covise settings for VS Code" << std::endl;
        return 0;
    }


    try
    {
        std::cerr << "parsing " << coviseSettingsFileName << std::endl;
        auto coviseSettings = json::parse(coviseSettingsFile, nullptr, true, true);
        coviseSettings.merge_patch(dynamicCoviseSettings);
        std::cerr << "parsing " << coviseLaunchFileName << std::endl;
        auto coviseLaunch = json::parse(coviseLaunchFile, nullptr, true, true); 

        if(userSettingsFile.is_open())  {
            std::cerr << "parsing " << userSettingsFileName << std::endl;
            auto userSettings = json::parse(userSettingsFile, nullptr, true, true);
            if(overwrite)
                userSettings.merge_patch(coviseSettings);
            else
                coviseSettings.merge_patch(userSettings);
        }
        std::ofstream userSettingsOut(userSettingsFileName);
        userSettingsOut << std::setw(4) << coviseSettings;

        if(!userLaunchFile.is_open())
        {
            std::ofstream userLaunchOut(userLaunchFileName);
            userLaunchOut << std::setw(4) << coviseLaunch;
        }
        else
        {
            std::cerr << "parsing " << userLaunchFileName << std::endl;
            auto userLaunch = json::parse(userLaunchFile, nullptr, true, true); 
            auto coviseConfigurations = coviseLaunch.find("configurations");
            if(coviseConfigurations == coviseLaunch.end())
                return 0;
            auto userConfigurations = userLaunch.find("configurations");
            if(userConfigurations == userLaunch.end())
                userLaunch.merge_patch(coviseLaunch);
            else
            {
                for(const auto &coviseConfig : *coviseConfigurations)
                {
                    bool userConfigExists = false;
                    for(const auto &userConfig : *userConfigurations)
                    {
                        if(userConfig["name"] == coviseConfig["name"])
                        {
                            userConfigExists = true;
                            break;
                        }
                    }
                    if(!userConfigExists)
                        userConfigurations->push_back(coviseConfig);
                }
            }
            std::ofstream userLaunchOut(userLaunchFileName);
            userLaunchOut << std::setw(4) << userLaunch;
        }
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    
}