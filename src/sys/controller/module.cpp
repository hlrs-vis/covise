/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "config.h"
#include "crb.h"
#include "exception.h"
#include "global.h"
#include "handler.h"
#include "host.h"
#include "module.h"
#include "renderModule.h"
#include "util.h"

#include <cassert>

using namespace covise;
using namespace covise::controller;

size_t NetModule::moduleCount = 1000;

NetModule::NetModule(const RemoteHost &host, const ModuleInfo &moduleInfo, int instance)
    : SubProcess(moduleType, host, moduleInfo.category == "Renderer" ? sender_type::RENDERER : sender_type::APPLICATIONMODULE, moduleInfo.name), m_info(moduleInfo), moduleId(moduleCount++)
{
    if (instance == -1)
    {
        ++moduleInfo.count;
        m_instance = moduleInfo.count;
    }
    else
    {
        m_instance = moduleInfo.count = instance;
    }
}

bool NetModule::isOnTop() const
{
    bool onTop = true;
    connectivity().forAllNetInterfaces([&onTop](const net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Input && interface.get_conn_state())
                                           {
                                               onTop = false;
                                           }
                                       });
    return onTop;
}

NetModule::~NetModule()
{
    for (NetModule *app : m_mirrors)
    {
        if (m_mirror != NOT_MIRR) //ist mirrored
        {
            app->m_mirror = NOT_MIRR;
            app->m_mirrors.clear();
        }
        if (auto renderer = dynamic_cast<Renderer *>(app))
        {
            if (m_mirror == CPY_MIRR)
            {
                try
                {
                    renderer->addDisplayAndHandleConnections(dynamic_cast<const Userinterface &>(host.getProcess(sender_type::USERINTERFACE)));
                }
                catch (const Exception &e)
                {
                    std::cerr << e.what() << '\n';
                }
            }
        }
    }
    // delete the netlinks
    auto it = netLinks.begin();
    while (it != netLinks.end())
    {
        it->del_link(info().name, std::to_string(instance()), getHost());
        it = netLinks.erase(it);
    }
    // del the corresponding to-Connection in the Object
    for (auto &interface : connectivity().interfaces)
    {
        if (interface->get_direction() == Direction::Input)
        {
            if (auto netInter = dynamic_cast<net_interface *>(interface.get()))
            {
                if (auto obj = netInter->get_object())
                    obj->del_to_connection(info().name, std::to_string(instance()), getHost(), interface->get_name());
            }
            else if (auto renderInter = dynamic_cast<render_interface *>(interface.get()))
            {
                renderInter->del_all_connections(this);
            }
        }
    }
}

void NetModule::resetId()
{
    moduleCount = 0;
}

const ModuleInfo &NetModule::info() const
{
    return m_info;
}

void NetModule::exec(NumRunning &numRunning)
{
    if (numRunning.apps == 0 || (!isOneRunningAbove(true)))
    {
        if (!isExecuting())
        {
            ++numRunning.apps;
            execute(numRunning);      //decreases numRunning on failure
            if (numRunning.apps == 1) // switch to execution mode
            {
                Message ex_msg{COVISE_MESSAGE_UI, "INEXEC"};
                host.hostManager.sendAll<Userinterface>(ex_msg);
                host.hostManager.sendAll<Renderer>(ex_msg);
            }

            if (m_mirror == ORG_MIRR)
            {
                for (NetModule *mirror : m_mirrors)
                {
                    mirror->execute(numRunning);
                }
            }
        }
        else
        {
            ++m_numRunning;
            Message err_msg{COVISE_MESSAGE_WARNING, "Controller\n \n \n Sorry: module is already running !"};
            host.hostManager.sendAll<Userinterface>(err_msg);
        }
    }
    else
    {
        Message err_msg{COVISE_MESSAGE_WARNING, "Controller\n \n \n Sorry: Is already one executing above!\n"};
        host.hostManager.sendAll<Userinterface>(err_msg);
    }
}

std::string NetModule::fullName() const
{
    return m_info.name + "_" + std::to_string(instance());
}

const std::string &NetModule::title() const
{
    if (m_title.empty())
        m_title = info().name + "_" + std::to_string(instance());
    return m_title;
}

void NetModule::setTitle(const std::string &t)
{
    m_title = t;
    std::stringstream ss;
    ss << "MODULE_TITLE\n"
       << info().name << "\n"
       << instance() << "\n"
       << getHost() << "\nSetModuleTitle\nString\n1\n"
       << title();
    Message msg{COVISE_MESSAGE_UI, ss.str()};
    send(&msg);
}

bool NetModule::startflag() const
{
    return m_isStarted;
}

void NetModule::resetStartFlag()
{
    m_isStarted = false;
}

void NetModule::setStartFlag()
{
    m_isStarted = true;
}

size_t NetModule::instance() const
{
    return m_instance;
}

void NetModule::init(const MapPosition &pos, int copy, ExecFlag flag, NetModule *mirror)
{
    m_position = pos;
    m_execFlag = flag;
    if (!start(std::to_string(m_instance).c_str(), info().category.c_str()) || !connectToCrb())
    {
        throw Exception{"Application::init failed to start module " + info().name};
    }
    if (copy == 4 && mirror)
        this->mirror(mirror);
    Message msg;
    recv_msg(&msg);
    if (!msg.data.data())
    {
        cerr << endl
             << " Application::init() - NULL module description received\n";
        exit(0);
    }
    m_info.readConnectivity(msg.data.data());
    //create_netlink() //links to the mirror modules
    initConnectivity();
    if (mirror && copy == 3) // node created by moving
    {
        m_instance = mirror->m_instance;
    }
}

void NetModule::mirror(NetModule *original)
{
    original->m_mirror = ORG_MIRR; //original
    original->m_mirrors.emplace_back(this);

    m_mirror = CPY_MIRR; //mirror
    m_mirrors.emplace_back(original);
}

void NetModule::initConnectivity()
{
    // copy predefined connectivity for this kind of module StaticModuleInfo
    copyConnectivity();
    int outNum = 0;
    for (auto &interface : m_connectivity.interfaces)
    {
        // generate data objectname for output port (ancient: DOCONN)
        net_interface *netInterface = dynamic_cast<net_interface *>(interface.get());
        if (interface->get_direction() == Direction::Output && netInterface)
        {
            ostringstream objName;
            objName << m_info.name << "_" << m_instance << "(" << moduleId << ")_OUT_" << outNum << "_";
            ++outNum;
            object *obj = CTRLGlobal::getInstance()->objectList->select(objName.str());
            if (!netInterface->get_conn_state() && !obj->check_from_connection(this, interface->get_name())) //no connection
            {
                netInterface->set_connect(obj);
                obj->connect_from(this, *netInterface);
            }
            else if (!netInterface->get_conn_state() || !obj->check_from_connection(this, interface->get_name()))
                print_comment(__LINE__, __FILE__, " ERROR: Connection between object and module destroyed!\n");
        }
    }
}

void NetModule::copyConnectivity()
{
    m_connectivity.inputParams = m_info.connectivity().inputParams;
    m_connectivity.outputParams = m_info.connectivity().outputParams;
    m_connectivity.interfaces.resize(m_info.connectivity().interfaces.size());
    std::transform(m_info.connectivity().interfaces.begin(), m_info.connectivity().interfaces.end(), m_connectivity.interfaces.begin(), [](const std::unique_ptr<C_interface> &inter)
                   { return std::unique_ptr<net_interface>(new net_interface{*inter}); });
}

int NetModule::testOriginalcount(const string &interfaceName) const
{
    if (m_mirror == ORG_MIRR)
    {
        // Original -> Zaehler 1 zurueckgeben
        return 1;
    }
    else
    {
        auto org = std::find_if(m_mirrors.begin(), m_mirrors.end(), [](const NetModule *mirror)
                                { return mirror->m_mirror == ORG_MIRR; });
        if (org != m_mirrors.end())
        {
            try
            {
                return (*org)->m_connectivity.getInterface<net_interface>(interfaceName).get_object()->get_counter();
            }
            catch (const Exception &e)
            {
                std::cerr << e.what() << '\n';
                return -1;
            }
        }
    }
    return -1;
}

const ModuleNetConnectivity &NetModule::connectivity() const
{
    return m_connectivity;
}

ModuleNetConnectivity &NetModule::connectivity()
{
    return m_connectivity;
}

const NetModule::MapPosition &NetModule::pos() const
{
    return m_position;
}

void NetModule::move(const NetModule::MapPosition &pos)
{
    m_position = pos;
}

std::string NetModule::createBasicModuleDescription() const
{
    std::stringstream ss;
    ss << info().name << "\n"
       << instance() << "\n"
       << getHost() << "\n";
    return ss.str();
}

std::string NetModule::createDescription() const
{
    std::stringstream ss;
    ss << info().name << "\n"
       << info().category << "\n"
       << getHost() << "\n"
       << info().description() << "\n";

    int numInputInterfaces = 0, numOutputInterfaces = 0;
    for (const auto &interface : m_connectivity.interfaces)
    {
        interface->get_direction() == controller::Direction::Input ? numInputInterfaces++ : numOutputInterfaces++;
    }
    ss << numInputInterfaces << "\n"
       << numOutputInterfaces << "\n"
       << m_connectivity.inputParams.size() << "\n"
       << m_connectivity.outputParams.size() << "\n";
    for (const auto &interface : m_connectivity.interfaces)
    {
        ss << interface->get_name() << "\n"
           << interface->get_type() << "\n"
           << interface->get_text() << "\n"
           << interface->get_demand() << "\n";
    }
    for (const parameter &param : m_connectivity.inputParams)
    {
        ss << param.getDescription();

        ss << param.get_extension() << "\n";
    }
    for (const parameter &param : m_connectivity.outputParams)
    {
        ss << param.getDescription();
    }
    return ss.str();
}

bool NetModule::isOriginal() const
{
    return m_mirror != CPY_MIRR;
}

bool NetModule::isExecuting() const
{
    return m_status == Status::executing;
}

void NetModule::setExecuting(bool state)
{
    m_status = state ? Status::executing : Status::Idle;
}

NetModule::Status NetModule::status() const
{
    return m_status;
}

void NetModule::setStatus(NetModule::Status status)
{
    m_status = status;
}

void NetModule::setAlive(bool state)
{
    m_alive = state;
}

std::vector<std::string> &NetModule::errorsSentByModule()
{
    return m_errorsSentByModule;
}

const std::vector<std::string> &NetModule::errorsSentByModule() const
{
    return m_errorsSentByModule;
}

void NetModule::set_DO_status(int mode, const string &DO_name)
{
    bool found = false;
    for (auto &interface : connectivity().interfaces)
    {
        if (interface->get_direction() == controller::Direction::Input)
        {
            if (auto appInterface = dynamic_cast<net_interface *>(interface.get()))
            {
                if (appInterface->get_conn_state())
                {
                    object *obj = appInterface->get_object();
                    if (obj->test(DO_name))
                    {
                        found = true;
                        obj->set_DO_status(DO_name, mode, *this, appInterface->get_name());
                    }
                }
            }
        }
    }
    if (found == false)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Dataobject not found \n");
        cerr << "\nERROR: Dataobject not found !!!\n";
    }
}

void NetModule::sendFinish()
{
    string content = get_outparaobj();
    if (!content.empty())
    {
        Message msg{COVISE_MESSAGE_FINISHED, content};
        host.hostManager.sendAll<Userinterface>(msg);
    }
}

void NetModule::delete_rez_objs()
{
    connectivity().forAllNetInterfaces([](net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output && interface.get_conn_state())
                                           {
                                               object *p_obj = interface.get_object();
                                               if (p_obj != NULL)
                                               {
                                                   p_obj->del_rez_DO();
                                                   p_obj->del_dep_data();
                                               }
                                           }
                                       });
}

void NetModule::onConnectionClosed()
{
    std::stringstream ss;
    ss << "DIED\n"
       << info().name << "\n"
       << instance() << "\n"
       << host.userInfo().ipAdress;
    Message msg{COVISE_MESSAGE_UI, ss.str()};
    host.hostManager.sendAll<Userinterface>(msg);

    ss = std::stringstream{};
    ss << "Module " << fullName() << "@" << host.userInfo().ipAdress << " crashed !!!";
    host.hostManager.sendAll<Userinterface>(Message{COVISE_MESSAGE_COVISE_ERROR, ss.str()});

    CTRLHandler::instance()->finishExecuteIfLastRunning(*this);
    setAlive(false);
}

std::string NetModule::getStartMessage()
{
    std::stringstream buff;
    buff << serialize();

    int numOutputConnections = 0;
    m_connectivity.forAllNetInterfaces([&numOutputConnections](const net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output || interface.get_conn_state())
                                           {
                                               ++numOutputConnections;
                                           }
                                       });
    buff << numOutputConnections << "\n"
         << m_connectivity.inputParams.size() << "\n";

    for (const auto &i : m_connectivity.interfaces)
    {
        if (auto ii = dynamic_cast<const net_interface *>(i.get()))
        {
            auto &interface = *ii;
            if (interface.get_conn_state())
            {
                const object *obj = interface.get_object();
                assert(obj);
                std::string obj_name = obj->get_current_name();
                if (obj_name.empty())
                {
                    return "";
                }

                buff << interface.get_name() << "\n"
                     << obj_name << "\n"
                     << obj->get_type() << "\n";

                if (obj->get_to().empty())
                {
                    buff << "UNCONNECTED"
                         << "\n";
                }
                else
                {
                    buff << "CONNECTED"
                         << "\n";
                }
            }
            else if (interface.get_direction() == controller::Direction::Output)
            {
                buff << interface.get_name() << "\nwrong_object_name\nwrong_object_type\n";
                string buf = "ERROR old Network file, replace module " + info().name;
                Message err_msg{COVISE_MESSAGE_COVISE_ERROR, buf};
                host.hostManager.sendAll<Userinterface>(err_msg);
                return "";
            }
            else if (interface.get_demand() == "req")
            {
                print_comment(__LINE__, __FILE__, "ERROR: get-startmessage. Interfaces not connected \n");
                std::stringstream msg;
                msg << "Warning: Required input port (" << interface.get_name() << ")" << info().name << "_" << moduleId << "@" << getHost();
                msg << " is not connected !! ";
                sendWarningMsgToMasterUi(msg.str());
                return "";
            }
        }
    }
    for (const parameter &param : m_connectivity.inputParams)
    {
        buff << param.serialize();
    }
    return buff.str();
}

void NetModule::sendWarningMsgToMasterUi(const std::string &msg)
{
    std::string data = "Controller\n \n \n" + msg;
    Message message{COVISE_MESSAGE_WARNING, data};
    for (const Userinterface *ui : host.hostManager.getAllModules<Userinterface>())
    {
        if (ui->status() == Userinterface::Master)
        {
            ui->send(&message);
        }
    }
}

bool NetModule::delete_old_objs()
{
    bool deleted = false;
    m_connectivity.forAllNetInterfaces([&deleted](net_interface &interface)
                                       {
                                           // is the Interface connected ?
                                           if (interface.get_direction() == Direction::Output && interface.get_conn_state())
                                           {
                                               object *p_obj = interface.get_object();
                                               if (p_obj && !p_obj->isEmpty())
                                               {
                                                   p_obj->del_old_DO();
                                                   deleted = true;
                                                   return;
                                               }
                                           }
                                       });
    return deleted;
}

void NetModule::new_obj_names()
{
    m_connectivity.forAllNetInterfaces([](net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output && interface.get_conn_state())
                                           {
                                               interface.get_object()->newDataObject();
                                           }
                                       });
}

std::string NetModule::get_outparaobj() const
{
    std::stringstream buff;
    buff << serialize();

    buff << m_connectivity.outputParams.size() << "\n";
    for (const auto &param : m_connectivity.outputParams)
    {
        buff << param.serialize();
    }

    int intf_count = getNumInterfaces(controller::Direction::Output);

    bool err = false;
    if (type == sender_type::RENDERER)
        buff << "0\n";
    else
    {
        buff << intf_count << "\n";

        // get objects
        m_connectivity.forAllNetInterfaces([&err, &buff](const net_interface &interface)
                                           {
                                               if (interface.get_direction() == controller::Direction::Output)
                                               {
                                                   if (interface.get_conn_state())
                                                   {
                                                       buff << interface.get_name() << "\n";

                                                       string obj_name = interface.get_object()->get_current_name();
                                                       if (!obj_name.empty())
                                                           buff << obj_name << "\n";
                                                       else
                                                           buff << "NO_OBJ\n"; //DeletedModuleFinished");
                                                   }
                                                   else
                                                   {
                                                       print_comment(__LINE__, __FILE__, "ERROR: send_finisCTRLGlobal::getInstance()-> Interfaces not connected \n");
                                                       cerr << "\n ERROR: send_finisCTRLGlobal::getInstance()-> Interfaces not connected !!!\n";
                                                       err = true;
                                                       return;
                                                   } // if state
                                               }
                                           });
    } // !renderer
    return err ? "" : buff.str();
}

std::string NetModule::get_inparaobj() const
{
    std::stringstream buffS;
    buffS << serialize();

    buffS << m_connectivity.inputParams.size() << "\n";
    for (const auto &param : m_connectivity.inputParams)
    {
        buffS << param.serialize();
    }

    buffS << getNumInterfaces(controller::Direction::Input) << "\n";

    m_connectivity.forAllNetInterfaces([&buffS, this](const net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Input)
                                           {
                                               buffS << serializeInputInterface(interface);
                                           }
                                       });
    return buffS.str();
}

bool NetModule::startModuleWaitingAbove(NumRunning &numRunning)
{
    bool oneWaiting = false;
    m_connectivity.forAllNetInterfaces([&numRunning, &oneWaiting, this](net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Input)
                                           {
                                               // is the interface connected?
                                               if (interface.get_conn_state() == true)
                                               {
                                                   auto mod = interface.get_object()->get_from().get_mod();
                                                   if (mod->startflag())
                                                   {
                                                       oneWaiting = true;
                                                       mod->resetStartFlag();
                                                       if (!mod->isOneRunningAbove(1))
                                                       {
                                                           numRunning.apps++;
                                                           mod->execute(numRunning);
                                                       }
                                                   }
                                               }
                                           }
                                       });
    return oneWaiting;
}

int NetModule::numRunning() const
{
    return m_numRunning;
}

void NetModule::setStart()
{
    connectivity().forAllNetInterfaces([](net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output)
                                           {
                                               interface.get_object()->setStartFlagOnConnectedModules();
                                           }
                                       });
}

void NetModule::startModulesUnder(NumRunning &numRunning)
{
    connectivity().forAllNetInterfaces([&numRunning](net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output)
                                           {
                                               interface.get_object()->start_modules(numRunning);
                                           }
                                       });
}

bool NetModule::isOneRunningAbove(bool first) const
{
    for (const auto &inter : m_connectivity.interfaces)
    {
        if (auto interface = dynamic_cast<const net_interface *>(inter.get()))
        {
            if (interface->get_direction() == controller::Direction::Input &&
                interface->get_conn_state() &&
                interface->get_object()->is_one_running_above())
            {
                return true;
            }
        }
    }
    if (!first)
    {
        if (m_isStarted || isExecuting())
        {
            return true;
        }
    }
    return false;
}

bool NetModule::is_one_running_under() const
{
    bool oneRunningUnder = false;
    connectivity().forAllNetInterfaces([&oneRunningUnder](const net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output &&
                                               interface.get_object())
                                           {
                                               const auto &to = interface.get_object()->get_to();
                                               for (const auto &conn : to)
                                               {
                                                   const NetModule *app = conn.get_mod();
                                                   if (app &&
                                                       !dynamic_cast<const Renderer *>(app) &&
                                                       app->isExecuting())
                                                   {
                                                       oneRunningUnder = true;
                                                       return;
                                                   }
                                               }
                                           }
                                       });
    return oneRunningUnder;
}

void NetModule::execute(NumRunning &numRunning)
{
    if (!m_alive)
        return;

    // set the Inputdataobjects to OLD
    // set status of Module to RUNNING
    m_numRunning = 0;

    if (is_one_running_under())
    {
        m_isStarted = true;
        numRunning.apps--;
        return;
    }
    setExecuting(true);

    //delete_all Objects if not saved
    bool ret = delete_old_objs();
    //give new Names to Output_objects
    new_obj_names();
    m_errorsSentByModule.clear();

    string content = getStartMessage();
    if (content.empty())
    {
        delete_rez_objs();
        if (ret)
            sendFinish();
        --numRunning.apps;
        setExecuting(false);
        string text = "Sorry, can't execute module " + info().name + "_" + std::to_string(instance()) + "@" + getHost();
        Message err_msg{COVISE_MESSAGE_WARNING, text};
        host.hostManager.sendAll<Userinterface>(err_msg);
        return;
    }

    Message msg{COVISE_MESSAGE_START, content};
    send(&msg);

    content = this->get_inparaobj();
    if (!content.empty())
    {
        msg = Message{COVISE_MESSAGE_START, content};
        host.hostManager.sendAll<Userinterface>(msg);
    }
}

int NetModule::overflowOfNextError() const
{
    constexpr int maxErrors = 25;
    return std::max(static_cast<int>(m_errorsSentByModule.size()) + 1 - maxErrors, 0);
}

int NetModule::numMirrors() const
{
    return m_mirrors.size();
}

std::vector<NetModule *> &NetModule::getMirrors()
{
    return m_mirrors;
}

void NetModule::delete_dep_objs()
{
    bool objs_deleted = false;

    m_connectivity.forAllNetInterfaces([&objs_deleted](net_interface &interface)
                                       {
                                           if (interface.get_direction() == controller::Direction::Output && interface.get_conn_state())
                                           {
                                               object *p_obj = interface.get_object();
                                               if ((p_obj != NULL) && (!p_obj->isEmpty()))
                                               {
                                                   p_obj->del_all_DO(0);
                                                   objs_deleted = true;
                                                   p_obj->del_dep_data();
                                               }
                                           }
                                       });
    // update the mapeditor with
    if (objs_deleted)
        sendFinish();
}

std::string NetModule::get_parameter(controller::Direction direction, bool forSaving) const
{
    const auto &params = direction == controller::Direction::Input ? connectivity().inputParams : connectivity().outputParams;
    int i = 0;
    std::stringstream ss;
    ss << params.size() << "\n";
    for (const auto &param : params)
    {
        ss << param.get_name() << "\n"
           << param.get_type() << "\n"
           << param.get_text() << "\n";
        string value = param.get_val_list();
        if (param.get_type() == "Browser" && forSaving)
        {
            auto fullPath = host.getProcess(sender_type::CRB).as<CRBModule>()->covisePath;
            string sep = fullPath.substr(0, 1);
            fullPath.erase(0, 1);
            auto pathList = splitStringAndRemoveComments(fullPath, sep);
            for (int i = 0; i < pathList.size(); i++)
            {
                const string &path = pathList[i];
                int find = (int)value.find(path);
                if (find != std::string::npos)
                {
                    value = value.substr(path.length());
                    while (value.length() > 0 && value[0] == '/')
                        value.erase(0, 1);
                    break;
                }
            }
        }
        ss << value << "\n"
           << "\n"
           << param.get_addvalue() << "\n";
    }
    return ss.str();
}

std::string NetModule::get_interfaces(controller::Direction direction) const
{
    std::stringstream buffer;
    int i = 0;
    for (const auto &interface : connectivity().interfaces)
    {
        if (interface->get_direction() == direction)
        {
            ++i;
            buffer << interface->get_name() << "\n"
                   << interface->get_type() << "\n"
                   << interface->get_text() << "\n"
                   << interface->get_demand() << "\n"
                   << "\n";
        }
    }
    return std::to_string(i) + "\n" + buffer.str();
}

std::string NetModule::get_moduleinfo() const
{
    stringstream buffer;
    buffer << info().name << "\n"
           << instance() << "\n"
           << (&host.hostManager.getLocalHost() == &host ? "LOCAL" : getHost()) << "\n"
           << info().category << "\n"
           << title() << "\n"
           << pos().x << "\n"
           << pos().y << "\n";
    return buffer.str();
}

std::string NetModule::get_module(bool forSaving) const
{
    stringstream buffer;
    buffer << "# Module " << info().name << "\n"
           << get_moduleinfo()
           << get_interfaces(controller::Direction::Input)
           << get_interfaces(controller::Direction::Output)
           << get_parameter(controller::Direction::Input, forSaving)
           << get_parameter(controller::Direction::Output, forSaving);

    return buffer.str();
}

void NetModule::setDeadFlag(int flag)
{
    connectivity().forAllNetInterfaces([flag](net_interface &interface)
                                       { interface.setDeadFlag(flag); });
}

void NetModule::writeScript(std::ofstream &of) const
{
    of << "#" << endl;
    of << "# MODULE: " << info().name << endl;
    of << "#" << endl;
    of << fullName() << " = " << info().name << "()" << endl;

    of << "network.add( " << fullName() << " )" << endl;

    // get Position x y
    of << fullName() << ".setPos( " << pos().x << ", " << pos().y << " )" << endl;

    of << "#" << endl;
    of << "# set parameter values" << endl;
    of << "#" << endl;

    for (const auto &param : connectivity().inputParams)
    {
        of << fullName() << ".set_" << param.get_name() << "( " << param.get_pyval_list() << " )" << endl;
    }
}

void NetModule::setObjectConn(const string &from_intf, object *obj)
{
    try
    {
        m_connectivity.getInterface<net_interface>(from_intf).set_connect(obj);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void NetModule::delObjectConn(const string &from_intf, object *obj)
{
    (void)obj;
    try
    {
        m_connectivity.getInterface<net_interface>(from_intf).del_connect();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

std::string NetModule::serialize() const
{
    std::stringstream buff;
    buff << info().name << "\n"
         << instance() << "\n"
         << host.userInfo().ipAdress << "\n";
    return buff.str();
}

size_t NetModule::getNumInterfaces(controller::Direction direction) const
{
    size_t count = 0;
    for (const auto &interface : m_connectivity.interfaces)
    {
        if (interface->get_direction() == direction)
        {
            ++count;
        }
    }
    return count;
}

std::string NetModule::serializeInputInterface(const net_interface &interface) const
{
    std::stringstream buff;
    if (interface.get_conn_state())
    {
        buff << interface.get_name() << "\n"
             << interface.get_object()->get_current_name() << "\n";
    }
    else
        buff << interface.get_name() << "\nNOT CONNECTED>\n";
    return buff.str();
}
