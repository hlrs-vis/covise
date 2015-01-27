/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:   COVISE FastDNAml FastDNAml module                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author: D.Rantzau, U. Woessner                                         **
 **                                                                        **
 **                                                                        **
 ** Date:  14.02.95  V1.0                                                  **
\**************************************************************************/

#include "FastDNAml.h"

#include <util/coviseCompat.h>
#include <sys/stat.h>

#include "newick/newick_tree.h"

int main(int argc, char *argv[])
{
    FastDNAml *application = new FastDNAml();
    application->start(argc, argv);
}

FastDNAml::FastDNAml()
    : coModule("Online connection to FastDNAml Simulation")
{
    count = 1;
    p_ivOut = addOutputPort("tree", "coDoText_Iv", "Iv Tree");
    master_conn = NULL;

    p_host = addStringParam("hostName", "MetaMaster Host");
    p_host->setImmediate(1);
    p_host->setValue("pcgpc7");

    p_port = addInt32Param("port", "MetaMaster Port");
    p_port->setImmediate(1);
    p_port->setValue(31011);

    p_path = addFileBrowserParam("path", "file path");
    p_path->setValue("/usr/local/data/test.tree", "*.trf;*.tree");
    p_path->setImmediate(1);

    p_getData = addBooleanParam("getData", "get new data");
    p_getData->setImmediate(1);
    p_getData->setValue(0);
    p_colors = addBooleanParam("colors", "use color coding");
    p_colors->setImmediate(1);
    p_colors->setValue(1);
    p_length = addBooleanParam("length", "use length encoding");
    p_length->setImmediate(1);
    p_length->setValue(1);
    p_names = addBooleanParam("names", "add name label");
    p_names->setImmediate(1);
    p_names->setValue(0);
    p_lowres = addBooleanParam("lowres", "display lowres version of the tree");
    p_lowres->setImmediate(1);
    p_lowres->setValue(0);
}

FastDNAml::~FastDNAml()
{
}

//
//
//..........................................................................
//
void FastDNAml::quit()
{
    //
    // ...... delete your data here .....
    //
}

void FastDNAml::param(const char *paramName)
{
    if (strcmp(paramName, p_path->getName()) == 0)
    {
        const char *Path = p_path->getValue();
        int doColor = true;
        int ignoreLength = false;

        doColor = p_colors->getValue();
        ignoreLength = !p_length->getValue();
        ivbuf = readTreeFile(Path, ignoreLength, doColor, p_names->getValue(), p_lowres->getValue(), -1);

        if (ivbuf == NULL)
        {
            char buf[1000];
            strcpy(buf, "ERROR: Can't access file :");
            strcat(buf, Path);
            Covise::sendError(buf);
            return;
        }
        selfExec();
    }
    if ((strcmp(paramName, p_port->getName()) == 0))
    {
        const char *hostName = p_host->getValue();
        const int port = p_port->getValue();
        cerr << "trying to connect to " << hostName << port << endl;

        // check if we have to reopen the conncetion to the specified server...

        if (master_conn == NULL)
        {
            master_conn = gui_new();
            master_conn->server_name = strdup(hostName);
            master_conn->port = (int)port;
        }

        if (master_conn->sock == -1 || master_conn->state == GUI_STATE_DISCONNECTED)
        {

            master_conn->sock = sock_connect((char *)hostName, (int)port);

            if (master_conn->sock >= 0)
            {

                master_conn->state = GUI_STATE_CONNECTED;

                // Now we have a master connection. First thing to receive is
                // the application list.

                master_conn->appl_list = gui_appl_list_read(master_conn->sock);
                cerr << "connected to " << hostName << port << endl;
            }
            else
                master_conn->state = GUI_STATE_DISCONNECTED;
        }

        // We know should have a valid connection if state == GUI_STATE_CONNECTED

        if (master_conn->state == GUI_STATE_CONNECTED)
        {
            gui_get_tree(master_conn);
        }
    }
    if ((strcmp(paramName, p_getData->getName()) == 0))
    {

        // We know should have a valid connection if state == GUI_STATE_CONNECTED

        if (master_conn && master_conn->state == GUI_STATE_CONNECTED)
        {
            gui_get_tree(master_conn);
        }
    }
}

void FastDNAml::postInst()
{
}

float FastDNAml::idle()
{
    // check for messages from Meta-Master
    // after receiving data
    //  selfExec();

    int doColor = true;
    int ignoreLength = false;
    int applID;

    gui_command_t *newCommand;

    if (master_conn != NULL && master_conn->state == GUI_STATE_CONNECTED)
    {

        newCommand = gui_command_wait(master_conn, 10);

        if (newCommand != NULL)
        {

            // gui_handle_command(master_conn, newCommand);

            switch (newCommand->command)
            {

            case APPL_COMMAND_NEW_APPLICATION:
            {
                appl_t *newAppl;
                int newApplID;
                cerr << "Got a new application." << std::endl;
                sock_read_int(master_conn->sock, &newApplID);
                newAppl = appl_read_info(master_conn->sock);
                gui_appl_list_add(master_conn->appl_list, newAppl, newApplID);
            }
            break;

            case APPL_COMMAND_REGISTER_DATA:
            {
                gui_appl_t *gappl;
                cerr << "Regoister  data" << std::endl;
                applID = newCommand->ID;
                cerr << "Appl " << newCommand->ID << endl;
                gappl = gui_appl_list_find(master_conn->appl_list, applID);
                if (gappl != NULL)
                {
                    appl_data_list_delete(&gappl->appl->data);
                    gappl->appl->data = appl_data_list_read(master_conn->sock);
                }
            }
            break;

            case APPL_COMMAND_APPLICATION_DISCONNECT:
                sock_read_int(master_conn->sock, &applID);
                gui_appl_list_remove(master_conn->appl_list, applID);
                break;

            case APPL_COMMAND_DATA_SET:
            {
                cerr << "receiving data " << endl;
                appl_data_descr_t *newDescr;
                appl_data_t *newData;
                appl_t *appl;
                gui_appl_t *gappl;

                newDescr = appl_data_descr_read(master_conn->sock);
                printf("Data will be named %s.\n", newDescr->name);
                gappl = gui_appl_list_find(master_conn->appl_list, newCommand->ID);
                if (gappl != NULL)
                {
                    newData = appl_data_list_find(gappl->appl->data, newDescr->name);
                    if (newData != NULL)
                    {
                        printf("Found data in application list.\n");
                        appl_data_read_ip(newData, master_conn->sock);

                        /* Action on data read goes here. */

                        char *treeStr = (char *)newData->repr;
                        ivbuf = readTree(ignoreLength, doColor, treeStr, p_names->getValue(), p_lowres->getValue(), newCommand->ID);
                        if (ivbuf)
                        {
                            printf("calling selfExec();\n");
                            selfExec();
                        }
                        free(treeStr);
                        newData->repr = NULL;
                    }
                }
                else
                {
                    printf("Application to be send is NOT registered.\n");
                }
            }
            break;
            }

            gui_command_delete(&newCommand);
        }
    }

    return 0.1; // poll a tenth of a second for covise messages
}

int FastDNAml::compute(const char *)
{
    //
    // ...... do work here ........
    //

    // read input parameters and data object name
    char *adress;
    int length;

    const char *IvDescr;
    IvDescr = p_ivOut->getObjName();
    descr = NULL;
    count++;

    cerr << "compute " << endl;

    if (IvDescr != NULL)
    {
        if (ivbuf != NULL)
        {
            length = strlen(ivbuf);
            descr = new coDoText(IvDescr, length);
            if (descr->objectOk())
            {
                descr->getAddress(&adress);
                memcpy(adress, ivbuf, length);
                ivbuf = NULL;
                p_ivOut->setCurrentObject(descr);
            }
            else
            {
                Covise::sendError("ERROR: object name not correct for 'Iv-File'");
                return FAIL;
            }
        }
        ivbuf = NULL;
    }
    else
    {
        Covise::sendError("ERROR: creation of data object 'descr' failed");
        return FAIL;
    }

    return SUCCESS;
}
