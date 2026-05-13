#pragma once

#include "BaseUI.h"
#include "app/system/EnergyType.h"
#include "app/system/Storage.h"
#include <lib/core/interfaces/ui/IComponent.h>
#include <lib/core/interfaces/ui/IButton.h>
#include <lib/core/interfaces/ui/ISelectionList.h>
#include <lib/core/interfaces/ui/IGUIFactory.h>

#include <memory>
#include <map>
#include <vector>
#include <string>

typedef std::vector<EnergyType> TypeRange;
typedef std::vector<Storage> StorageRange;

struct SimulationUIConfig
{
    const core::interface::ui::IGUIFactory &factory;
    core::interface::ui::IComponent *parent;
    std::string name;
    TypeRange type_range;
    StorageRange storage_range;
};

class SimulationUI : BaseUI
{
public:
    SimulationUI(SimulationUIConfig config);
    void setLiftCallback(const std::function<void(bool)> &func) { m_lift->setCallback(func); }
    void setOnGridTypeChanged(const std::function<void(int)> &func) { m_energyGrids->setCallback(func); }
    void setOnScenarioChanged(std::function<void(int)> cb) { m_scenarios->setCallback(cb); }
    void setOnStorageChanged(EnergyType type, const std::function<void(int)> &func);
    void setStorageDefault(EnergyType type, Storage storage);

    void setScenarioList(const std::vector<std::string> &names);
    void setStorageList(const std::map<EnergyType, std::vector<std::string>> &names);
    void setGridList(const std::vector<std::string> &names);
    Storage getSelectedStorage(EnergyType type);

private:
    void init(SimulationUIConfig &config);
    void initGrids(const core::interface::ui::IGUIFactory &factory, const TypeRange& range);
    void initStorage(const core::interface::ui::IGUIFactory &factory, const TypeRange& typeRange, const StorageRange &storageRange);
    void initSelectionList(core::interface::ui::ISelectionList* selList, const std::vector<std::string> &names);

    SimulationUIConfig m_config;
    std::unique_ptr<core::interface::ui::IMenu> m_menu;
    std::unique_ptr<core::interface::ui::ISelectionList> m_energyGrids;
    std::unique_ptr<core::interface::ui::ISelectionList> m_scenarios;
    std::unique_ptr<core::interface::ui::IButton> m_lift;
    std::map<EnergyType, std::unique_ptr<core::interface::ui::ISelectionList>> m_storage;
};
