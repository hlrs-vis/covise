/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_CONTROL_PROCESS_H
#define CTRL_CONTROL_PROCESS_H

#include <covise/covise_process.h>
#include <shm/covise_shm.h>
#include "config.h"
namespace covise
{
namespace controller{


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

public:

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
    void handle_shm_msg(Message *msg) override;

    /**
       *  getting the access to the shared memory
       *
       *  @param   conn    reference to the infos of the connection
       *                   towards the datamanager
       */
    void get_shared_memory(const covise::Connection *conn);

    /**
       *  getting the address of the shared memory
       */
    void *get_shared_memory_adress()
    {
        return shm->get_pointer();
    };
    // nur temporaer eingefuegt, zum testen !! ^
};
} //namespace controller
} //namespace covise
#endif
