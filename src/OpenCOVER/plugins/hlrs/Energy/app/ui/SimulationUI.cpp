#include "SimulationUI.h"
#include "BaseUI.h"
#include "app/system/EnergyType.h"
#include "app/system/Storage.h"
#include <stdexcept>
#include <algorithm>

using namespace core::interface::ui;

SimulationUI::SimulationUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent)
    : BaseUI(name, parent)
{
    init(factory, name, parent);
}

void SimulationUI::init(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent)
{
    if (!parent)
        throw std::runtime_error("SimulationUI cannot be initialized properly because the parent is NULL.");

    m_menu = factory.createMenu("Simulation", parent);

    m_lift = factory.createButton(m_menu.get(), "Up");

    m_energyGrids = factory.createSelectionList(m_menu.get(), "Grid Selection");
    std::vector<std::string> typeNames(ENERGYTYPE_RANGE.size());
    std::transform(ENERGYTYPE_RANGE.begin(), ENERGYTYPE_RANGE.end(), typeNames.data(), [&](EnergyType type)
        { return EnergyTypeToString(type); });
    m_energyGrids->setList(typeNames);
    
    std::vector<std::string> storageNames(FULL_STORAGE_RANGE.size());
    std::transform(FULL_STORAGE_RANGE.begin(), FULL_STORAGE_RANGE.end(), storageNames.data(), [&](Storage storage)
        { return StorageToString(storage); });
    
    std::string selListName("");
    for (auto type: ENERGYTYPE_RANGE){
        selListName = EnergyTypeToString(type);
        auto selList = factory.createSelectionList(m_menu.get(), selListName + " Storage");
        selList->setList(storageNames);
        m_storage[type] = std::move(selList);
    }
}

void SimulationUI::setStorageSelectionCallback(EnergyType type, const std::function<void(int)> &func)
{
    if (auto it = m_storage.find(type); it != m_storage.end())
        it->second->setCallback(func);
}

void SimulationUI::setStorageSelectionDefault(EnergyType type, Storage storage) 
{
    if (auto it = m_storage.find(type); it != m_storage.end())
    {
       auto &selList = it->second; 
       auto storage_ptr = std::find(FULL_STORAGE_RANGE.begin(), FULL_STORAGE_RANGE.end(), storage);
       int index_of_ptr = storage_ptr - FULL_STORAGE_RANGE.begin();
       selList->select(index_of_ptr);
    }
}

Storage SimulationUI::getSelectedStorage(EnergyType type)
{
    if (m_storage.find(type) == m_storage.end())
        return Storage::UNKNOWN;
    // will work because of the initialization order in init()
    return FULL_STORAGE_RANGE[m_storage[type]->selectedIndex()];
}
