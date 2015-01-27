/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

class covise__Event
{
public:
    std::string type 1; ///< Required element.

    covise__Event(const std::string &type);
    virtual ~covise__Event();

    virtual covise__Event *clone() const;
};

class covise__LinkAddEvent : public covise__Event
{
public:
    covise__Link link 1; ///< Required element.

    covise__LinkAddEvent(const covise__Link &link);
    virtual ~covise__LinkAddEvent();

    virtual covise__Event *clone() const;
};

class covise__LinkDelEvent : public covise__Event
{
public:
    std::string linkID 1; ///< Required element.

    covise__LinkDelEvent(const std::string &linkID);
    virtual ~covise__LinkDelEvent();

    virtual covise__Event *clone() const;
};

class covise__ModuleAddEvent : public covise__Event
{
public:
    covise__Module module 1; ///< Required element.

    covise__ModuleAddEvent(const covise__Module &module);
    virtual ~covise__ModuleAddEvent();

    virtual covise__Event *clone() const;
};

class covise__ModuleDelEvent : public covise__Event
{
public:
    std::string moduleID 1; ///< Required element.

    covise__ModuleDelEvent(const std::string &moduleID);
    virtual ~covise__ModuleDelEvent();

    virtual covise__Event *clone() const;
};

class covise__ModuleChangeEvent : public covise__Event
{
public:
    covise__Module module 1; ///< Required element.

    covise__ModuleChangeEvent(const covise__Module &module);
    virtual ~covise__ModuleChangeEvent();

    virtual covise__Event *clone() const;
};

class covise__ModuleDiedEvent : public covise__Event
{
public:
    std::string moduleID 1; ///< Required element.

    covise__ModuleDiedEvent(const std::string &moduleID);
    virtual ~covise__ModuleDiedEvent();

    virtual covise__Event *clone() const;
};

class covise__ModuleExecuteStartEvent : public covise__Event
{
public:
    std::string moduleID 1; ///< Required element.

    covise__ModuleExecuteStartEvent(const std::string &moduleID);
    virtual ~covise__ModuleExecuteStartEvent();

    virtual covise__Event *clone() const;
};

class covise__ModuleExecuteFinishEvent : public covise__Event
{
public:
    std::string moduleID 1; ///< Required element.

    covise__ModuleExecuteFinishEvent(const std::string &moduleID);
    virtual ~covise__ModuleExecuteFinishEvent();

    virtual covise__Event *clone() const;
};

class covise__ExecuteStartEvent : public covise__Event
{
public:
    covise__ExecuteStartEvent();
    virtual ~covise__ExecuteStartEvent();

    virtual covise__Event *clone() const;
};

class covise__ExecuteFinishEvent : public covise__Event
{
public:
    covise__ExecuteFinishEvent();
    virtual ~covise__ExecuteFinishEvent();

    virtual covise__Event *clone() const;
};

class covise__ParameterChangeEvent : public covise__Event
{
public:
    std::string moduleID 1; ///< Required element.
    covise__Parameter *parameter 1; ///< Required element.

    covise__ParameterChangeEvent(const std::string &moduleID, const covise__Parameter *parameter);
    covise__ParameterChangeEvent(const covise__ParameterChangeEvent &parameter);
    virtual ~covise__ParameterChangeEvent();

    virtual covise__Event *clone() const;
};

class covise__OpenNetEvent : public covise__Event
{
public:
    std::string mapname 1; ///< Required element.

    covise__OpenNetEvent(const std::string &mapname);
    virtual ~covise__OpenNetEvent();

    virtual covise__Event *clone() const;
};

class covise__OpenNetDoneEvent : public covise__Event
{
public:
    std::string mapname 1; ///< Required element.

    covise__OpenNetDoneEvent(const std::string &mapname);
    virtual ~covise__OpenNetDoneEvent();

    virtual covise__Event *clone() const;
};

class covise__QuitEvent : public covise__Event
{
public:
    covise__QuitEvent();
    virtual ~covise__QuitEvent();

    virtual covise__Event *clone() const;
};
