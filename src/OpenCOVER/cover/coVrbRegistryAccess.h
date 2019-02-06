///* This file is part of COVISE.
//
//   You can use it under the terms of the GNU Lesser General Public License
//   version 2.1 or later, see lgpl-2.1.txt.
//
// * License: LGPL 2+ */
//
//#ifndef CO_VRB_REGISTRY_ACCESS_H
//#define CO_VRB_REGISTRY_ACCESS_H
//
///*! \file
// \brief interface class for application modules to the VRB registry
//
// \author Dirk Rantzau
// \author (C) 1998
//         Computer Centre University of Stuttgart,
//         Allmandring 30,
//         D-70550 Stuttgart,
//         Germany
//
// \date   15.07.1998
// */
//
//#include <util/coTypes.h>
//#include <net/tokenbuffer.h>
//#include <util/coDLList.h>
//#include <string>
//#include <map>
//
//namespace opencover
//{
//class coVrbRegistryAccess;
//class coVrbRegObserver;
//class coVrbRegEntryObserver;
//
//class COVEREXPORT coVrbRegObserverType
//{
//
//public:
//    coVrbRegObserverType()
//    {
//    }
//    ~coVrbRegObserverType()
//    {
//    }
//
//    enum RegType
//    {
//        UnknownObserverType,
//        CO_REG_CHANGED_OBSERVER,
//        // This has to be the last
//        numObserverTypes
//    };
//};
//
//class COVEREXPORT coVrbRegEntry
//{
//
//public:
//    coVrbRegEntry(const char *clName, int ID, const char *varName = NULL);
//    virtual ~coVrbRegEntry();
//    int getID(void)
//    {
//        return _ID;
//    }
//    bool isDeleted()
//    {
//        return _isDeleted;
//    }
//    const char *getVar(void)
//    {
//        return _var.c_str();
//    }
//    const char *getValue(void) const
//    {
//        return _val.get_data();
//    }
//	covise::TokenBuffer& getData();
//    const char *getClass(void)
//    {
//        return _cl.c_str();
//    }
//    void setID(int id)
//    {
//        _ID = id;
//    };
//    void setVar(const char *var)
//    {
//        _var = var;
//    }
//    void setVal(const char *val)
//    {
//		_val.reset();
//		_val<< val;
//    }
//    void setVal(covise::TokenBuffer &&tb)
//    {
//		_val = std::move(tb);
//	}
//    bool isClassOnlyEntry()
//    {
//        return _isClassOnly;
//    }
//    virtual void attach(coVrbRegEntryObserver *);
//    virtual void detach(coVrbRegEntryObserver *);
//    virtual void notify(int interestType);
//    void changedByMe(bool byMe = true)
//    {
//        _changedByMe = byMe;
//    };
//    void setChanged()
//    {
//        notify(coVrbRegObserverType::CO_REG_CHANGED_OBSERVER);
//    }
//    void setDeleted()
//    {
//        _isDeleted = true;
//        notify(coVrbRegObserverType::CO_REG_CHANGED_OBSERVER);
//    }
//    /**
//       *  if a VRB connected, resend local variables and subscriptions.
//       */
//    void updateVRB();
//    void setValue(const char *val);
//	void setData(covise::TokenBuffer &&tb);
//private:
//    coVrbRegEntryObserver *_observer;
//    int _ID;
//    std::string _cl;
//    std::string _var;
//    covise::TokenBuffer _val;
//    bool _isClassOnly;
//    bool _isDeleted;
//    bool _changedByMe;
//};
//
///**
// * Registry access cl
// * @author Dirk Rantzau
// * @date 15.07.98
// *
// */
//class COVEREXPORT coVrbRegistryAccess
//{
//
//public:
//    static coVrbRegistryAccess *instance;
//    /// construct a registry access path to the controller
//    coVrbRegistryAccess(int id);
//    void setID(int id);
//
//    /**
//       *  Subscribe to all variables in a registry cl
//       *
//       *  @cl    registry cl
//       *  @ID       module ID of interest, 0 for all
//       *  @ob       observer cl to be attached for updates
//       */
//    void subscribeClass(const char *clName, int ID, coVrbRegEntryObserver *ob);
//
//    /**
//       *  Subscribe to a specific variable of a registry cl
//       *
//       *  @cl    registry cl
//       *  @ID       module ID of interest, 0 for all
//       *  @var      variable in registry cl
//       *  @ob       observer cl to be attached for updates
//       */
//    coVrbRegEntry *subscribeVar(const char *cl, int ID, const char *var, const covise::TokenBuffer &value, coVrbRegEntryObserver *ob);
//
//    /**
//       *  Unsubscribe from a registry cl (previously subscribed with subscribecl)
//       *
//       *  @cl    registry cl
//       *  @ID       module ID of interest, 0 for all
//       *
//       */
//    void unsubscribeClass(const char *cl, int ID);
//
//    /**
//       *  Unsubscribe from a specific variable of a registry cl
//       *
//       *  @cl      registry cl
//       *  @ID         module ID of interest, 0 for all
//       *  @var        registry variable belonging to the cl
//       *
//       */
//    void unsubscribeVar(const char *cl, int ID, const char *var);
//
//    /**
//       *  Create a specific cl variable in the registry. If the cl
//       *  variable is already existing the operation is ignored. If the cl
//       *  for the given variable is not existing, the cl is created in the registry.
//       *
//       *  @cl  registry cl
//       *  @var    registry variable belonging to the cl
//       *  @flag   flag=0: session local variable, flag=1: global variable surviving a session
//       */
//    void createVar(const char *cl, const char *var, int flag = 0);
//
//    /**
//       *  Sets a specific variable value in the registry. All observers attached
//       *  to this variable will be notified immediately about the change.
//       *
//       *  @cl  registry cl
//       *  @var    registry variable belonging to the cl
//       *  @val    current variable value to be set in the registry
//       */
//    void setVar(const char *cl, const char *var, const char *val);
//
//    /**
//       *  Destroys a specific variable in the registry. All observers attached
//       *  to this variable will be notified immediately about deletion.
//       *  Note: It is normally not necessary to destroy variables no longer used!
//       *
//       *  @cl  registry cl
//       *  @var    registry variable belonging to a cl
//       */
//    void destroyVar(const char *cl, const char *var);
//
//    /**
//       *  if a VRB connected, resend local variables and subscriptions.
//       */
//    void updateVRB();
//
//    void update(covise::TokenBuffer &tb, int reason);
//
//    virtual ~coVrbRegistryAccess();
//    void sendMsg(covise::TokenBuffer &tb, int message_type);
//
//    int getID()
//    {
//        return _ID;
//    };
//
//protected:
//private:
//    void addEntry(coVrbRegEntry *e);
//    void removeEntry(coVrbRegEntry *e);
//
//    int _ID;
//    std::string _name;
//    covise::coDLPtrList<coVrbRegEntry *> *_entryList;
//};
//
//class COVEREXPORT coVrbRegObserver
//{
//
//public:
//    virtual ~coVrbRegObserver()
//    {
//    }
//    virtual void update(coVrbRegEntry *theChangedRegEntry) = 0;
//    virtual int getObserverType() = 0;
//
//protected:
//    coVrbRegObserver()
//    {
//    }
//};
//
//class COVEREXPORT coVrbRegEntryObserver : public coVrbRegObserver
//{
//
//public:
//    virtual ~coVrbRegEntryObserver(){};
//    virtual void update(coVrbRegEntry *theChangedRegEntry) = 0;
//    int getObserverType()
//    {
//        return coVrbRegObserverType::CO_REG_CHANGED_OBSERVER;
//    }
//
//protected:
//    coVrbRegEntryObserver()
//    {
//    }
//};
//}
//#endif // _CO_REGISTRY_ACCESS_H_
