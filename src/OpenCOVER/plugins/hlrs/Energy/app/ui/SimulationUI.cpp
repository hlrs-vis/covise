#include "SimulationUI.h"
#include "BaseUI.h"
#include "app/system/EnergyType.h"
#include "app/system/Storage.h"
#include <stdexcept>
#include <algorithm>

using namespace core::interface::ui;

SimulationUI::SimulationUI(SimulationUIConfig config)
    : BaseUI(config.name, config.parent)
    , m_config(config)
{
    init(config);
}

void SimulationUI::init(SimulationUIConfig &config)
{
    if (!config.parent)
        throw std::runtime_error("SimulationUI cannot be initialized properly because the parent is NULL.");

    m_menu = config.factory.createMenu(config.name, config.parent);

    m_lift = config.factory.createButton(m_menu.get(), "Up");

    initGrids(config.factory, config.type_range);
    // m_energyGrids = config.factory.createSelectionList(m_menu.get(), "Grid Selection");
    initStorage(config.factory, config.type_range, config.storage_range);

    // Create the scenario selection list widget
    m_scenarios = config.factory.createSelectionList(m_menu.get(), "Scenarios");
}

void SimulationUI::initGrids(const core::interface::ui::IGUIFactory &factory, const TypeRange& range)
{
    if (!m_menu)
        throw std::runtime_error("Menu not properly initizialized");
    m_energyGrids = factory.createSelectionList(m_menu.get(), "Grid Selection");
    std::vector<std::string> typeNames(range.size());
    std::transform(range.begin(), range.end(), typeNames.data(), [&](EnergyType type)
        { return EnergyTypeToString(type); });
    setGridList(typeNames);
    // m_energyGrids->setList(typeNames);
}

void SimulationUI::initStorage(const core::interface::ui::IGUIFactory &factory, const TypeRange& typeRange, const StorageRange &storageRange)
{
    if (!m_menu)
        throw std::runtime_error("Menu not properly initizialized");
    std::vector<std::string> storageNames(storageRange.size());
    std::transform(storageRange.begin(), storageRange.end(), storageNames.data(), [&](Storage storage)
        { return StorageToString(storage); });

    for (auto type : typeRange)
    {
        std::string selListName = std::string(EnergyTypeToString(type)) + " Storage";
        auto selList = factory.createSelectionList(m_menu.get(), selListName);
        selList->setList(storageNames);
        m_storage[type] = std::move(selList);
    }
}

void SimulationUI::setScenarioList(const std::vector<std::string> &names)
{
    initSelectionList(m_scenarios.get(), names);
}

void SimulationUI::setStorageList(const std::map<EnergyType, std::vector<std::string>> &names) 
{
   for (auto &[type, storagenames]: names) {
        if (m_storage.find(type) == m_storage.end()) {
            std::string selListName = std::string(EnergyTypeToString(type)) + " Storage";
            m_storage[type] = m_config.factory.createSelectionList(m_menu.get(), selListName);
        }
        initSelectionList(m_storage[type].get(), storagenames);
   }
}

void SimulationUI::setGridList(const std::vector<std::string> &names) {
    initSelectionList(m_energyGrids.get(), names);
}

void SimulationUI::initSelectionList(core::interface::ui::ISelectionList* selList, const std::vector<std::string> &names) {
    if(selList)
    {
        if (!names.empty()) {
            selList->setList(names);
            selList->select(0);
        }
    }
}

void SimulationUI::setOnStorageChanged(EnergyType type, const std::function<void(int)> &func)
{
    if (auto it = m_storage.find(type); it != m_storage.end())
        it->second->setCallback(func);
}

void SimulationUI::setStorageDefault(EnergyType type, Storage storage)
{
    if (auto it = m_storage.find(type); it != m_storage.end())
    {
        auto &selList = it->second;
        auto storage_ptr = std::find(FULL_STORAGE_RANGE.begin(), FULL_STORAGE_RANGE.end(), storage);
        int index_of_ptr = static_cast<int>(storage_ptr - FULL_STORAGE_RANGE.begin());
        selList->select(index_of_ptr);
    }
}

Storage SimulationUI::getSelectedStorage(EnergyType type)
{
    if (m_storage.find(type) == m_storage.end())
        return Storage::UNKNOWN;

    return m_config.storage_range[m_storage[type]->selectedIndex()];
}
