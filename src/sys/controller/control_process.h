/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_PROCESS_H
#define CTRL_PROCESS_H

#include <covise/covise_process.h>
#include <shm/covise_shm.h>
#include "covise_module.h"

namespace covise
{
   enum class ExecFlag : int;
   
/**
 *  Controller class:  contains the informations
 *  concerning the controller process
 */

class Controller : public Process // process for the controller
{

    /// pointer to the sharedmemory object
    ShmAccess *shm;
      /**
       *  when starting a renderer ask crb if a plugin is possible and get the right name
       *  @param   category    the category of the applicationmodule
       *  @param   dmod        the object with de infos about the conn to dtm.
       */                                     
    bool confirmIsRenderer(const char* cat, AppModule* dmod);
protected:
    /// used numeration of application modules
    int module_count;

public:
    /**
       *  start datamanager on the local host
       *  @param   name  the name of the datamanager executable
       *  @return  object holding information about the connection to dtm.
       */
    AppModule *start_datamanager(const string &name);

    /**
       *  start datamanager on the remote host
       *  @param   rhost   the remote host
       *  @param   name    the name of the datamanager executable
       *  @return  object holding information about the connection to dtm.
       */
    AppModule *start_datamanager(Host *rhost, const char *name);

    /**
       *  start datamanager on the remote host
       *  @param   rhost     the remote host
       *  @param   user      the username of the remote account
       *  @param   passwd    the password of the remote account
       *  @param   name      the name of the datamanager executable
       *  @return  object holding information about the connection to dtm.
       */
    AppModule *start_datamanager(Host *rhost, const char *user, const char *passwd, const char *name);

    /**
       *  start datamanager on the remote host
       *  @param   rhost       the remote host
       *  @param   user        the username of the remote account
       *  @param   name        the name of the datamanager executable
       *  @param   exec_type   the method used for starting datamanager
       *  @param   script_name the name of the script file used
       *                       for starting dtm.
       *  @return  object holding information about the connection to dtm.
       */
    AppModule *start_datamanager(Host *rhost, const char *user, const char *name, int exec_type, const char *script_name = NULL);

    
    /**
       *  starting an application module
       *  @param   peer_type   type of the started module
       *  @param   name        the name of the applicationmodule
       *  @param   dmod        the object with de infos about the conn to dtm.
       *  @param   instance    the id number of the application module
       *  @param   category    the category of the applicationmodule
       *  @param   param       additional parameters for applicationmodule
       *  @return  object holding info about the connection
       *           to the application module
       */
    AppModule *start_applicationmodule(sender_type peer_type, const char *name, AppModule *dmod, const char *instance, 
                                       ExecFlag flags, const char *category = nullptr, const std::vector<std::string> &params = std::vector<std::string>{});



    /**
       *  create the object handling the informations
       *  concerning the controller process
       *  @param   name        the name of the process
       */
    Controller(const char *name)
        : // initialization
        Process(name, 1000, CONTROLLER)
    {
        shm = NULL;
        module_count = 1000; // Controller ==  1000
    };

    /**
       *  create the object handling the informations
       *  concerning the controller process
       *  @param   name        the name of the process
       *  @param   port        the open port of the process
       */
    Controller(const char *name, int port)
        : // initialization
        Process(name, 1000, CONTROLLER, port)
    {
        shm = NULL;
        module_count = 1000; // SimpleController ==  1000
    };

    /// destructor
    ~Controller()
    {
        //	print_delete(__LINE__, __FILE__,list_of_connections);
        //	delete list_of_connections;
        delete shm;
    }; // destructor

    /**
       *  adding new segments to the shared memory
       *
       *  @param   msg    the message containing infos about the
       *                  new segment needed
       */
    void handle_shm_msg(Message *msg);

    /**
       *  getting the access to the shared memory
       *
       *  @param   dmod    reference to the infos of the connection
       *                   towards the datamanager
       */
    void get_shared_memory(AppModule *dmod);

    void addConnection(Connection *conn);

    /**
       *  getting the address of the shared memory
       */
    void *get_shared_memory_adress()
    {
        return shm->get_pointer();
    };
    // nur temporaer eingefuegt, zum testen !! ^
};
}
#endif
