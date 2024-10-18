
#include "MathExpressions.h"
#include "Tool.h"
#include "ToolChanger/ToolChanger.h"
#include "VrmlNode.h"

#include <OpcUaClient/opcua.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>


class Machine : public LogicInterface
{
public:
    Machine(opencover::ui::Menu *menu, opencover::config::File *file, MachineNodeBase *node);
    void update() override;



private:
    bool m_rdy = false;
    MachineNodeBase *m_machineNode = nullptr;
    opencover::opcua::Client *m_client = nullptr;
    std::vector<opencover::opcua::ObserverHandle> m_valueIds;
    size_t m_index = 0;
    std::unique_ptr<SelfDeletingTool> m_tool;
    opencover::ui::Menu *m_menu = nullptr;
    opencover::ui::Button *m_pauseBtn = nullptr;
    opencover::config::File *m_configFile = nullptr;
    std::unique_ptr<MathExpressionObserver> m_mathExpressionObserver;
    std::vector<MathExpressionObserver::ObserverHandle::ptr> m_axisValueHandles;
    bool m_pauseMove = false;

    void move(int axis, float value);
    bool arrayMode() const;
    void pause(bool state);
    bool addTool();
    void connectOpcua();
    bool updateMachine(bool haveTool);

};
