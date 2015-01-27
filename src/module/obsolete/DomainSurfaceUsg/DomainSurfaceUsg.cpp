/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE Shows surface polygons of unstructured grids       **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Oliver Heck                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  22.03.96  V1.0                                                  **
\**************************************************************************/

#include "DomainSurfaceUsg.h"
#include <util/coviseCompat.h>
//
// Covise include stuff
//
//
//  functions

//  global variables
//
//  Shared memory data

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();

    return 0;
}

// =======================================================================
// START WORK HERE (main block)
// =======================================================================

int Application::add_vertex(int v)
{
    if (conn_tag[v] >= 0)
        return (conn_tag[v]);
    conn_tag[v] = num_vert;
    x_out[num_vert] = x_in[v];
    y_out[num_vert] = y_in[v];
    z_out[num_vert] = z_in[v];
    if (DataType == DATA_S)
    {
        u_out[num_vert] = u_in[v];
    }
    if (DataType == DATA_V)
    {
        u_out[num_vert] = u_in[v];
        v_out[num_vert] = v_in[v];
        w_out[num_vert] = w_in[v];
    }
    num_vert++;
    return (num_vert - 1);
};
int Application::ladd_vertex(int v)
{
    if (conn_tag[v] >= 0)
        return (conn_tag[v]);
    conn_tag[v] = lnum_vert;
    lx_out[lnum_vert] = x_out[v];
    ly_out[lnum_vert] = y_out[v];
    lz_out[lnum_vert] = z_out[v];

    if (DataType == DATA_S)
    {
        lu_out[lnum_vert] = u_out[v];
    }
    if (DataType == DATA_V)
    {
        lu_out[lnum_vert] = u_out[v];
        lv_out[lnum_vert] = v_out[v];
        lw_out[lnum_vert] = w_out[v];
    }

    lnum_vert++;
    return (lnum_vert - 1);
};

Application::Application(int argc, char *argv[])
    : d_title(NULL)
{
    Covise::set_param_callback(Application::paramCallback, this);
    Covise::set_module_description("Extract the Surface of an unstructured Grid");
    Covise::add_port(INPUT_PORT, "meshIn", "UnstructuredGrid", "Unstructured Grid");
    Covise::add_port(INPUT_PORT, "dataIn",
                     "Float|Vec3|Float|Vec3", "input data");
    Covise::add_port(OUTPUT_PORT, "meshOut", "Polygons", "Domain Surface");
    Covise::add_port(OUTPUT_PORT, "dataOut", "Float|Vec3",
                     "output data on vertices of surface polygons");
    Covise::add_port(OUTPUT_PORT, "linesOut", "Lines", "Boundary lines");
    Covise::add_port(OUTPUT_PORT, "ldataOut", "Float|Vec3",
                     "output data on vertices of boundary lines");
    Covise::add_port(PARIN, "angle", "FloatScalar", "Feature angle");
    Covise::add_port(PARIN, "vertex", "FloatVector", "Normal for backface culling");
    Covise::add_port(PARIN, "scalar", "FloatScalar", "Threshold for backface culling");
    Covise::add_port(PARIN, "double", "Boolean", "Double-Point check");

    Covise::add_port(PARIN, "optimize", "Choice", "should we care 'bout RAM or not");
    Covise::set_port_default("optimize", "1 speed memory");

    Covise::set_port_default("angle", "0.1");
    Covise::set_port_default("vertex", "1.0 0.0 0.0");
    Covise::set_port_default("scalar", "1.5");
    //	Covise::set_port_default("double","FALSE");
    Covise::set_port_default("double", "TRUE");
    Covise::set_port_required("dataIn", 0);
    Covise::set_port_dependency("dataOut", "dep dataIn");
    Covise::set_port_dependency("ldataOut", "dep dataIn");

    Covise::init(argc, argv);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::send_ui_message("MODULE_DESC", "Extract the Surface of an unstructured Grid");
}

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//=====================================================================
// test if surface should be displayed
//=====================================================================
inline int Application::test(int v1, int v2, int v3)
{
    float l, n1x, n1y, n1z;
    n1x = ((y_in[v1] - y_in[v2]) * (z_in[v1] - z_in[v3])) - ((z_in[v1] - z_in[v2]) * (y_in[v1] - y_in[v3]));
    n1y = ((z_in[v1] - z_in[v2]) * (x_in[v1] - x_in[v3])) - ((x_in[v1] - x_in[v2]) * (z_in[v1] - z_in[v3]));
    n1z = ((x_in[v1] - x_in[v2]) * (y_in[v1] - y_in[v3])) - ((y_in[v1] - y_in[v2]) * (x_in[v1] - x_in[v3]));
    l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
    if (l != 0)
    {
        n1x /= l;
        n1y /= l;
        n1z /= l;
        return ((n1x * n2x + n1y * n2y + n1z * n2z) < angle);
    }
    else
        return 0;
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================

void Application::doModule(coDistributedObject *meshIn,
                           coDistributedObject *dataIn,
                           char *meshOutName,
                           char *dataOutName,
                           char *lineOutName,
                           char *ldataOutName,
                           coDistributedObject **meshOut,
                           coDistributedObject **dataOut,
                           coDistributedObject **lineOut,
                           coDistributedObject **ldataOut,
                           int masterLevel = 0)
{

    char colorn[255];
    char *color_attr;
    const char *dtype = NULL;
    int data_anz = 0;

    conn_list = conn_tag = elem_list = lconn_list = lelem_list = NULL;
    x_out = y_out = z_out = lx_out = ly_out = lz_out = NULL;

    *ldataOut = *meshOut = *dataOut = *lineOut = NULL;
    int num_attr;
    const char **attr_n, **attr_v;

    //////////////// handle sets ////////////////////

    if (strcmp(meshIn->getType(), "SETELE") == 0)
    {
        if (dataIn && (strcmp(meshIn->getType(), "SETELE")))
        {
            Covise::sendError("ERROR: Data/Mesh Set levels garbled");
            meshOut = dataOut = lineOut = ldataOut = NULL;
            return;
        }

        ///////// Mask and space for Internal Mesh Object
        coDoSet *meshSet = (coDoSet *)meshIn;
        int meshSetLen;
        coDistributedObject *const *meshSetObj = meshSet->getAllElements(&meshSetLen);
        char *meshOutMask = new char[strlen(meshOutName) + 6];
        char *meshOutInt = new char[strlen(meshOutName) + 6];
        sprintf(meshOutMask, "%s_%%d", meshOutName);

        ///////// Mask and space for Internal Line Object
        char *lineOutMask = new char[strlen(lineOutName) + 6];
        char *lineOutInt = new char[strlen(lineOutName) + 6];
        sprintf(lineOutMask, "%s_%%d", lineOutName);

        if (dataIn) ///// with data
        {

            ///////// Mask and space for Internal Mesh Object
            coDoSet *dataSet = (coDoSet *)dataIn;
            int dataSetLen;
            coDistributedObject *const *dataSetObj = dataSet->getAllElements(&dataSetLen);
            char *dataOutMask = new char[strlen(dataOutName) + 6];
            char *dataOutInt = new char[strlen(dataOutName) + 6];
            sprintf(dataOutMask, "%s_%%d", dataOutName);

            char *ldataOutMask = new char[strlen(ldataOutName) + 6];
            char *ldataOutInt = new char[strlen(ldataOutName) + 6];
            sprintf(ldataOutMask, "%s_%%d", ldataOutName);

            if (dataSetLen == meshSetLen)
            {

                int i;
                coDistributedObject **meshOutObj
                    = new coDistributedObject *[meshSetLen + 1];
                meshOutObj[meshSetLen] = NULL;
                coDistributedObject **lineOutObj
                    = new coDistributedObject *[meshSetLen + 1];
                lineOutObj[meshSetLen] = NULL;
                coDistributedObject **dataOutObj
                    = new coDistributedObject *[meshSetLen + 1];
                dataOutObj[meshSetLen] = NULL;
                coDistributedObject **ldataOutObj
                    = new coDistributedObject *[meshSetLen + 1];
                ldataOutObj[meshSetLen] = NULL;

                int nullc = 0; // NULL counter

                for (i = 0; i < meshSetLen; i++)
                {
                    if (masterLevel)
                    {
                        char buffer[64];
                        sprintf(buffer, "Step %d", i);
                        Covise::sendInfo(buffer);
                    }
                    sprintf(meshOutInt, meshOutMask, i);
                    sprintf(lineOutInt, lineOutMask, i);
                    sprintf(dataOutInt, dataOutMask, i);
                    sprintf(ldataOutInt, ldataOutMask, i);
                    doModule(meshSetObj[i], dataSetObj[i],
                             meshOutInt, dataOutInt, lineOutInt, ldataOutInt,
                             &meshOutObj[i], &dataOutObj[i], &lineOutObj[i], &ldataOutObj[i]);

                    if (meshOutObj[i] == NULL && dataOutObj[i] == NULL)
                        nullc++;
                }

                // reduction of set
                // (necessary if the mesh contains exclusively bar elements => mesh, data are NULL)
                int length = meshSetLen - nullc; // reduced length of set
                coDistributedObject **redMeshOutObj;
                coDistributedObject **redDataOutObj;
                if (nullc != 0)
                {
                    int j = 0;
                    redMeshOutObj = new coDistributedObject *[length + 1];
                    redMeshOutObj[length] = NULL;
                    redDataOutObj = new coDistributedObject *[length + 1];
                    redDataOutObj[length] = NULL;
                    for (i = 0; i < length; i++)
                    {
                        while (meshOutObj[i + j] == NULL && dataOutObj[i + j] == NULL && i + j < meshSetLen)
                            j++;
                        redMeshOutObj[i] = meshOutObj[i + j];
                        redDataOutObj[i] = dataOutObj[i + j];
                    }
                }

                if (nullc == 0)
                {
                    *meshOut = new coDoSet(meshOutName, meshOutObj);
                    if (dataOut)
                        *dataOut = new coDoSet(dataOutName, dataOutObj);
                    *lineOut = new coDoSet(lineOutName, lineOutObj);
                    if (ldataOut)
                        *ldataOut = new coDoSet(ldataOutName, ldataOutObj);
                }
                else
                {
                    *meshOut = new coDoSet(meshOutName, redMeshOutObj);
                    if (dataOut)
                        *dataOut = new coDoSet(dataOutName, redDataOutObj);
                    *lineOut = new coDoSet(lineOutName, lineOutObj);
                    if (ldataOut)
                        *ldataOut = new coDoSet(ldataOutName, ldataOutObj);

                    delete[] redMeshOutObj;
                    delete[] redDataOutObj;
                }

                // setting attributes
                /*
            char *times=meshIn->getAttribute("TIMESTEP");
            if (times) {
            (*meshOut)->addAttribute("TIMESTEP", times);
            (*lineOut)->addAttribute("TIMESTEP", times);
            if(dataOut)
            (*dataOut)->addAttribute("TIMESTEP", times);
                  if (ldataOut)
            (*ldataOut)->addAttribute("TIMESTEP", times);
            }
            times=meshIn->getAttribute("READ_MODULE");
            if (times) {
            (*meshOut)->addAttribute("READ_MODULE", times);
            (*lineOut)->addAttribute("READ_MODULE", times);
            if(dataOut)
            (*dataOut)->addAttribute("READ_MODULE", times);
            if (ldataOut)
            (*ldataOut)->addAttribute("READ_MODULE", times);
            }
            times=meshIn->getAttribute("BLOCKINFO");
            if (times) {
            (*meshOut)->addAttribute("BLOCKINFO", times);
            (*lineOut)->addAttribute("BLOCKINFO", times);
            if(dataOut)
            (*dataOut)->addAttribute("BLOCKINFO", times);
            if (ldataOut)
            (*ldataOut)->addAttribute("BLOCKINFO", times);
            }
            times=meshIn->getAttribute("PART");
            if (times) {
            (*meshOut)->addAttribute("PART", times);
            (*lineOut)->addAttribute("PART", times);
            if(dataOut)
            (*dataOut)->addAttribute("PART", times);
            if (ldataOut)
            (*ldataOut)->addAttribute("PART", times);
            }
            */
                if (meshIn->getAttribute("Probe2D") == NULL)
                {
                    (*meshOut)->copyAllAttributes(meshIn);
                }
                else // update Probe2D attribute
                {
                    num_attr = (meshIn)->getAllAttributes(&attr_n, &attr_v);
                    for (i = 0; i < num_attr; i++)
                    {
                        if (strcmp(attr_n[i], "Probe2D") != 0)
                        {
                            (*meshOut)->addAttribute(attr_n[i], attr_v[i]);
                        }
                    }
                }

                (*meshOut)->addAttribute("Probe2D", dataOutName);

                (*lineOut)->copyAllAttributes(meshIn);
                if (dataOut)
                    (*dataOut)->copyAllAttributes(dataIn);
                if (ldataOut)
                    (*ldataOut)->copyAllAttributes(dataIn);

                // free
                for (i = 0; i < meshSetLen; i++)
                {
                    delete meshOutObj[i];
                    delete dataOutObj[i];
                    delete lineOutObj[i];
                    delete ldataOutObj[i];
                }
            }
            else
            {
                Covise::sendError("ERROR: Data/Mesh Set sizes garbled");
                meshOut = dataOut = lineOut = ldataOut = NULL;
            }
            delete[] dataOutMask;
            delete[] dataOutInt;
            delete[] ldataOutMask;
            delete[] ldataOutInt;
        }
        else ///// without data
        {

            int i;
            coDistributedObject **meshOutObj
                = new coDistributedObject *[meshSetLen + 1];
            meshOutObj[meshSetLen] = NULL;
            coDistributedObject **lineOutObj
                = new coDistributedObject *[meshSetLen + 1];
            lineOutObj[meshSetLen] = NULL;
            coDistributedObject *dummy_1;
            coDistributedObject *dummy_2;

            int nullc = 0; // NULL counter

            for (i = 0; i < meshSetLen; i++)
            {
                if (masterLevel)
                {
                    char buffer[64];
                    sprintf(buffer, "Step %d", i);
                    Covise::sendInfo(buffer);
                }
                sprintf(meshOutInt, meshOutMask, i);
                sprintf(lineOutInt, lineOutMask, i);
                doModule(meshSetObj[i], NULL,
                         meshOutInt, NULL, lineOutInt, NULL,
                         &meshOutObj[i], &dummy_1, &lineOutObj[i], &dummy_2);

                if (meshOutObj[i] == NULL)
                    nullc++;
            }

            // reduction of set
            // (necessary if the mesh contains exclusively bar elements => mesh is NULL)
            int length = meshSetLen - nullc; // reduced lenghth of set
            coDistributedObject **redMeshOutObj;

            if (nullc != 0)
            {
                int j = 0;
                redMeshOutObj = new coDistributedObject *[length + 1];
                redMeshOutObj[length] = NULL;
                for (i = 0; i < length; i++)
                {
                    while (meshOutObj[i + j] == NULL && i + j < meshSetLen)
                        j++;
                    redMeshOutObj[i] = meshOutObj[i + j];
                }
            }

            if (nullc == 0)
            {
                *meshOut = new coDoSet(meshOutName, meshOutObj);
                *lineOut = new coDoSet(lineOutName, lineOutObj);
            }
            else
            {
                *meshOut = new coDoSet(meshOutName, redMeshOutObj);
                *lineOut = new coDoSet(lineOutName, lineOutObj);

                delete[] redMeshOutObj;
            }

            // setting attributes
            (*meshOut)->copyAllAttributes(meshIn);
            (*lineOut)->copyAllAttributes(meshIn);
            if (dataOut)
                (*dataOut)->copyAllAttributes(dataIn);
            if (ldataOut)
                (*ldataOut)->copyAllAttributes(dataIn);

            /*
         char *times=meshIn->getAttribute("TIMESTEP");
         if (times) {
         (*meshOut)->addAttribute("TIMESTEP", times);
         (*lineOut)->addAttribute("TIMESTEP", times);
         }
         times=meshIn->getAttribute("READ_MODULE");
         if (times) {
         (*meshOut)->addAttribute("READ_MODULE", times);
         (*lineOut)->addAttribute("READ_MODULE", times);
         }
         times=meshIn->getAttribute("BLOCKINFO");
         if (times) {
         (*meshOut)->addAttribute("BLOCKINFO", times);
         (*lineOut)->addAttribute("BLOCKINFO", times);
         }
         times=meshIn->getAttribute("PART");
         if (times) {
         (*meshOut)->addAttribute("PART", times);
         (*lineOut)->addAttribute("PART", times);
         }
         */

            // free
            for (i = 0; i < meshSetLen; i++)
            {
                delete meshOutObj[i];
                delete lineOutObj[i];
            }
        }

        delete[] meshOutMask;
        delete[] meshOutInt;
        delete[] lineOutMask;
        delete[] lineOutInt;

        return;
    }

    //////////////// handle non-sets ////////////////////

    tmp_grid = (coDoUnstructuredGrid *)meshIn;
    if (strcmp(tmp_grid->getType(), "UNSGRD"))
    {
        Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
        return;
    }
    tmp_grid->getGridSize(&numelem, &numconn, &numcoord);
    tmp_grid->getAddresses(&el, &cl, &x_in, &y_in, &z_in);

    tmp_grid->getTypeList(&tl); // tl-Type List

    DataType = 0;
    if (dataIn)
    {

        dtype = dataIn->getType();
        if (strcmp(dtype, "USTSDT") == 0)
        {
            USIn = (coDoFloat *)dataIn;
            data_anz = USIn->getNumPoints();
            USIn->getAddress(&u_in);
            DataType = DATA_S;
            if (data_anz == numelem)
                DataType = DATA_S_E;
        }
        else if (strcmp(dtype, "USTVDT") == 0)
        {
            UVIn = (coDoVec3 *)dataIn;
            data_anz = UVIn->getNumPoints();
            UVIn->getAddresses(&u_in, &v_in, &w_in);
            DataType = DATA_V;
            if (data_anz == numelem)
                DataType = DATA_V_E;
        }
        else
        {
            Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
            return;
        }
    }

    if (DataType && (data_anz != numcoord) && (data_anz != numelem))
    {
        if (data_anz != 0)
            Covise::sendWarning("WARNING: Data objects dimension does not match grid ones: dummy output");
        DataType = DATA_NONE;
    }

    //	set color in geometry

    if ((color_attr = tmp_grid->getAttribute("COLOR")) == NULL)
    {
        strcpy(colorn, "White");
        color_attr = colorn;
    }

    //	Is there data in the array ?
    //      sl: if data and grid are dummy, there is no reason to alarm the gentle user
    if (DataType && data_anz != 0 && (numelem == 0 || numconn == 0 || numcoord == 0))
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    //      If computation is for the first time or the grid has changed
    //      create adjacency information
    //	get cells_use_coord list from shared memory

    tmp_grid->getNeighborList(&cuc_count, &cuc, &cuc_pos);
    numelem_o = numelem;

    u_out = v_out = w_out = 0;
    lu_out = lv_out = lw_out = 0;

    // Surface polygons
    surface();
    Polygons = new coDoPolygons(meshOutName, num_vert, x_out, y_out, z_out,
                                num_conn, conn_list, num_elem, elem_list);
    if (!meshIn->getAttribute("COLOR")) // sonst koennten wir COLOR
        // 2-mal schreiben (siehe unten
        // Polygons->copyAllAttributes(meshIn));
        Polygons->addAttribute("COLOR", color_attr);
    Polygons->addAttribute("vertexOrder", "2");

    *meshOut = Polygons;

    // Contour lines
    lines();
    tmp_grid->free_neighbor_list();
    Lines = new coDoLines(lineOutName, lnum_vert, lx_out, ly_out, lz_out,
                          lnum_conn, lconn_list, lnum_elem, lelem_list);
    if (!meshIn->getAttribute("COLOR"))
        Lines->addAttribute("COLOR", color_attr);
    Lines->addAttribute("vertexOrder", "2");
    *lineOut = Lines;

    // Data
    if (DataType == DATA_S)
    {
        if (num_vert != 0)
            SOut = new coDoFloat(dataOutName, num_vert, u_out);
        else
            SOut = new coDoFloat(dataOutName, 0);
        *dataOut = SOut;

        if (lnum_vert != 0)
            SlinesOut = new coDoFloat(ldataOutName, lnum_vert, lu_out);
        else
            SlinesOut = new coDoFloat(ldataOutName, 0);
        *ldataOut = SlinesOut;

        delete[] u_out;
        delete[] lu_out;
    }
    else if (DataType == DATA_V)
    {
        if (num_vert != 0)
            VOut = new coDoVec3(dataOutName, num_vert, u_out, v_out, w_out);
        else
            VOut = new coDoVec3(dataOutName, 0);
        *dataOut = VOut;

        if (lnum_vert != 0)
            VlinesOut = new coDoVec3(ldataOutName, lnum_vert, lu_out, lv_out, lw_out);
        else
            VlinesOut = new coDoVec3(ldataOutName, 0);
        *ldataOut = VlinesOut;

        delete[] u_out;
        delete[] v_out;
        delete[] w_out;

        delete[] lu_out;
        delete[] lv_out;
        delete[] lw_out;
    }
    else if (DataType == DATA_S_E)
    {
        if (num_elem)
            SOut = new coDoFloat(dataOutName, num_elem, u_out);
        else
            SOut = new coDoFloat(dataOutName, 0);
        *dataOut = SOut;

        if (lnum_elem)
            SlinesOut = new coDoFloat(ldataOutName, lnum_elem, lu_out);
        else
            SlinesOut = new coDoFloat(ldataOutName, 0);
        *ldataOut = SlinesOut;

        delete[] u_out;
        delete[] lu_out;
    }
    else if (DataType == DATA_V_E)
    {
        if (num_elem)
            VOut = new coDoVec3(dataOutName, num_elem, u_out, v_out, w_out);
        else
            VOut = new coDoVec3(dataOutName, 0);
        *dataOut = VOut;

        if (lnum_elem)
            VlinesOut = new coDoVec3(ldataOutName, lnum_elem, lu_out, lv_out, lw_out);
        else
            VlinesOut = new coDoVec3(ldataOutName, 0);
        *ldataOut = VlinesOut;

        delete[] u_out;
        delete[] v_out;
        delete[] w_out;

        delete[] lu_out;
        delete[] lv_out;
        delete[] lw_out;
    }
    else
    {
        if (dataIn && (strcmp(dtype, "STRSDT") == 0 || strcmp(dtype, "USTSDT") == 0))
        {
            *dataOut = new coDoFloat(dataOutName, 0);
            *ldataOut = new coDoFloat(ldataOutName, 0);
        }
        else if (dataIn && (strcmp(dtype, "STRVDT") == 0 || strcmp(dtype, "USTVDT") == 0))
        {
            *dataOut = new coDoVec3(dataOutName, 0);
            *ldataOut = new coDoVec3(ldataOutName, 0);
        }
        else
        {
            *dataOut = NULL;
            *ldataOut = NULL;
        }
    }

    // setting of attributes
    if (meshIn->getAttribute("Probe2D") == NULL)
    {
        Polygons->copyAllAttributes(meshIn);
    }
    else // update Probe2D attribute
    {
        num_attr = (meshIn)->getAllAttributes(&attr_n, &attr_v);
        for (int i = 0; i < num_attr; i++)
        {
            if (strcmp(attr_n[i], "Probe2D") != 0)
            {
                Polygons->addAttribute(attr_n[i], attr_v[i]);
            }
        }
    }

    Polygons->addAttribute("Probe2D", dataOutName);

    Lines->copyAllAttributes(meshIn);
    if (dataOut && *dataOut)
        (*dataOut)->copyAllAttributes(dataIn);
    if (ldataOut && *ldataOut)
        (*ldataOut)->copyAllAttributes(dataIn);

    /*
   char *times=meshIn->getAttribute("TIMESTEP");
   if (times) {
    Polygons->addAttribute("TIMESTEP", times);
    Lines->addAttribute("TIMESTEP", times);
    if (dataOut)
      if(*dataOut)
   (*dataOut)->addAttribute("TIMESTEP", times);
    if (ldataOut)
      if (*ldataOut)
   (*ldataOut)->addAttribute("TIMESTEP", times);
   }
   times=meshIn->getAttribute("READ_MODULE");
   if (times) {
   Polygons->addAttribute("READ_MODULE", times);
   Lines->addAttribute("READ_MODULE", times);
   if (dataOut)
   if(*dataOut)
   (*dataOut)->addAttribute("READ_MODULE", times);
   if (ldataOut)
   if(ldataOut)
   (*ldataOut)->addAttribute("READ_MODULE", times);
   }
   times=meshIn->getAttribute("BLOCKINFO");
   if (times) {
   Polygons->addAttribute("BLOCKINFO", times);
   Lines->addAttribute("BLOCKINFO", times);
   if (dataOut)
   if(*dataOut)
   (*dataOut)->addAttribute("BLOCKINFO", times);
   if (ldataOut)
   if(*ldataOut)
   (*ldataOut)->addAttribute("BLOCKINFO", times);
   }
   times=meshIn->getAttribute("PART");
   if (times) {
   Polygons->addAttribute("PART", times);
   Lines->addAttribute("PART", times);
   if (dataOut)
   if(*dataOut)
   (*dataOut)->addAttribute("PART", times);
   if (ldataOut)
   if(*ldataOut)
   (*ldataOut)->addAttribute("PART", times);
   }
   */

    delete[] x_out;
    delete[] y_out;
    delete[] z_out;
    delete[] conn_list;
    delete[] conn_tag;
    delete[] elem_list;
    delete[] lx_out;
    delete[] ly_out;
    delete[] lz_out;
    delete[] lconn_list;
    delete[] lelem_list;
    delete[] elemMap;
}

//////////////////////////////////////////////////////////////////////////////

void Application::lines()
{
    int i, j, np, n;
    int v1, v2, v3, v4, v21, v22, v23;
    float n1x, n1y, n1z, n2x, n2y, n2z, ang, l;
    lx_out = new float[num_vert + num_bar * 2];
    ly_out = new float[num_vert + num_bar * 2];
    lz_out = new float[num_vert + num_bar * 2];

    if (DataType == DATA_S)
    {
        lu_out = new float[num_vert + num_bar * 2];
    }
    else if (DataType == DATA_V)
    {
        lu_out = new float[num_vert + num_bar * 2];
        lv_out = new float[num_vert + num_bar * 2];
        lw_out = new float[num_vert + num_bar * 2];
    }
    else if (DataType == DATA_S_E)
    {
        lu_out = new float[num_elem * 6 + num_bar];
    }
    else if (DataType == DATA_V_E)
    {
        lu_out = new float[num_elem * 6 + num_bar];
        lv_out = new float[num_elem * 6 + num_bar];
        lw_out = new float[num_elem * 6 + num_bar];
    }

    Polygons->computeNeighborList();

    int el_alloc, cl_alloc;
    if (MEMORY_OPTIMIZED == 1)
    {

        lnum_vert = 0;
        lnum_conn = 0;
        lnum_elem = 0;
        el_alloc = cl_alloc = 0;

        for (i = 0; i < num_elem; i++)
        {
            if (i == num_elem - 1)
                np = num_conn - elem_list[i];
            else
                np = elem_list[i + 1] - elem_list[i];
            switch (np)
            {
            case 3:
            {
                v1 = conn_list[elem_list[i]];
                v2 = conn_list[elem_list[i] + 1];
                v3 = conn_list[elem_list[i] + 2];
                n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
                n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
                n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
                l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
                n1x /= l;
                n1y /= l;
                n1z /= l;
                if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
                if ((n = Polygons->getNeighbor(i, v2, v3)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
                if ((n = Polygons->getNeighbor(i, v1, v3)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
            }
            break;

            case 4:
            {
                v1 = conn_list[elem_list[i]];
                v2 = conn_list[elem_list[i] + 1];
                v3 = conn_list[elem_list[i] + 2];
                v4 = conn_list[elem_list[i] + 3];
                n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
                n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
                n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
                l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
                n1x /= l;
                n1y /= l;
                n1z /= l;
                if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
                if ((n = Polygons->getNeighbor(i, v2, v3)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
                if ((n = Polygons->getNeighbor(i, v3, v4)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
                if ((n = Polygons->getNeighbor(i, v4, v1)) >= 0)
                {
                    v21 = conn_list[elem_list[n]];
                    v22 = conn_list[elem_list[n] + 1];
                    v23 = conn_list[elem_list[n] + 2];
                    n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                    n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                    n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                    l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                    n2x /= l;
                    n2y /= l;
                    n2z /= l;
                    ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                    if (ang < 0)
                        ang = -ang;
                    ang = 1 - ang;
                    if (ang > tresh)
                    {
                        el_alloc++;
                        cl_alloc += 2;
                    }
                }
                else
                {
                    el_alloc++;
                    cl_alloc += 2;
                }
            }
            break;
            };
        }
        // handle the lines
        el_alloc += (num_bar);
        cl_alloc += (num_bar)*2;

        // that's it
    }
    else
    { // unoptimized
        el_alloc = 4 * num_elem + num_bar;
        cl_alloc = 8 * num_elem + 2 * num_bar;
    }

    lconn_list = new int[cl_alloc];
    lelem_list = new int[el_alloc];
    memset(conn_tag, -1, numcoord * sizeof(int));

    lnum_vert = 0;
    lnum_conn = 0;
    lnum_elem = 0;

    // cerr <<"TEST: " <<  num_elem << "\t" << num_bar << endl;

    for (i = 0; i < num_elem; i++)
    {
        if (i == num_elem - 1)
            np = num_conn - elem_list[i];
        else
            np = elem_list[i + 1] - elem_list[i];
        switch (np)
        {
        case 3:
        {
            v1 = conn_list[elem_list[i]];
            v2 = conn_list[elem_list[i] + 1];
            v3 = conn_list[elem_list[i] + 2];
            n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
            n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
            n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
            l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
            n1x /= l;
            n1y /= l;
            n1z /= l;
            if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v1);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v2);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v1);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v2);
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v2, v3)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v2);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v3);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v2);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v3);
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v1, v3)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v3);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v1);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v3);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v1);
                lnum_conn++;
            }
        }
        break;

        case 4:
        {
            v1 = conn_list[elem_list[i]];
            v2 = conn_list[elem_list[i] + 1];
            v3 = conn_list[elem_list[i] + 2];
            v4 = conn_list[elem_list[i] + 3];
            n1x = ((y_out[v1] - y_out[v2]) * (z_out[v1] - z_out[v3])) - ((z_out[v1] - z_out[v2]) * (y_out[v1] - y_out[v3]));
            n1y = ((z_out[v1] - z_out[v2]) * (x_out[v1] - x_out[v3])) - ((x_out[v1] - x_out[v2]) * (z_out[v1] - z_out[v3]));
            n1z = ((x_out[v1] - x_out[v2]) * (y_out[v1] - y_out[v3])) - ((y_out[v1] - y_out[v2]) * (x_out[v1] - x_out[v3]));
            l = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
            n1x /= l;
            n1y /= l;
            n1z /= l;
            if ((n = Polygons->getNeighbor(i, v1, v2)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v1);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v2);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v1);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v2);
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v2, v3)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v2);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v3);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v2);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v3);
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v3, v4)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v3);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v4);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v3);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v4);
                lnum_conn++;
            }
            if ((n = Polygons->getNeighbor(i, v4, v1)) >= 0)
            {
                v21 = conn_list[elem_list[n]];
                v22 = conn_list[elem_list[n] + 1];
                v23 = conn_list[elem_list[n] + 2];
                n2x = ((y_out[v21] - y_out[v22]) * (z_out[v21] - z_out[v23])) - ((z_out[v21] - z_out[v22]) * (y_out[v21] - y_out[v23]));
                n2y = ((z_out[v21] - z_out[v22]) * (x_out[v21] - x_out[v23])) - ((x_out[v21] - x_out[v22]) * (z_out[v21] - z_out[v23]));
                n2z = ((x_out[v21] - x_out[v22]) * (y_out[v21] - y_out[v23])) - ((y_out[v21] - y_out[v22]) * (x_out[v21] - x_out[v23]));
                l = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
                n2x /= l;
                n2y /= l;
                n2z /= l;
                ang = (n1x * n2x + n1y * n2y + n1z * n2z);
                if (ang < 0)
                    ang = -ang;
                ang = 1 - ang;
                if (ang > tresh)
                {
                    if (DataType == DATA_S_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        lu_out[lnum_elem] = u_out[i];
                        lv_out[lnum_elem] = v_out[i];
                        lw_out[lnum_elem] = w_out[i];
                    }
                    lelem_list[lnum_elem] = lnum_conn;
                    lnum_elem++;
                    lconn_list[lnum_conn] = ladd_vertex(v4);
                    lnum_conn++;
                    lconn_list[lnum_conn] = ladd_vertex(v1);
                    lnum_conn++;
                }
            }
            else
            {
                if (DataType == DATA_S_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                }
                else if (DataType == DATA_V_E)
                {
                    lu_out[lnum_elem] = u_out[i];
                    lv_out[lnum_elem] = v_out[i];
                    lw_out[lnum_elem] = w_out[i];
                }
                lelem_list[lnum_elem] = lnum_conn;
                lnum_elem++;
                lconn_list[lnum_conn] = ladd_vertex(v4);
                lnum_conn++;
                lconn_list[lnum_conn] = ladd_vertex(v1);
                lnum_conn++;
            }
        }
        break;
        };
    }

    for (i = 0; i < num_bar; i++)
    {
        lelem_list[lnum_elem] = lnum_conn;
        if (DataType == DATA_S_E)
        {
            lu_out[lnum_elem] = u_in[elemMap[i]];
        }
        else if (DataType == DATA_V_E)
        {
            lu_out[lnum_elem] = u_in[elemMap[i]];
            lv_out[lnum_elem] = v_in[elemMap[i]];
            lw_out[lnum_elem] = w_in[elemMap[i]];
        }
        lnum_elem++;
        // entries directly from grid lists
        for (j = 0; j < 2; j++)
        {

            lconn_list[lnum_conn] = lnum_vert;
            lnum_conn++;

            if (elemMap[i] != -1)
            {

                lx_out[lnum_vert] = x_in[cl[el[elemMap[i]] + j]];
                ly_out[lnum_vert] = y_in[cl[el[elemMap[i]] + j]];
                lz_out[lnum_vert] = z_in[cl[el[elemMap[i]] + j]];

                if (DataType == DATA_S)
                {
                    lu_out[lnum_vert] = u_in[cl[el[elemMap[i]] + j]];
                }
                else if (DataType == DATA_V)
                {
                    lu_out[lnum_vert] = u_in[cl[el[elemMap[i]] + j]];
                    lv_out[lnum_vert] = v_in[cl[el[elemMap[i]] + j]];
                    lw_out[lnum_vert] = w_in[cl[el[elemMap[i]] + j]];
                }

                lnum_vert++;
            }
            else
            {
                Covise::sendError("ERROR in elemMap");
            }
        }
    }
}

//=====================================================================
// create the surface of a domain
//=====================================================================
void Application::surface()
{
    int i, a, c;
    int nb; // ne deleted

    x_out = new float[numcoord];
    y_out = new float[numcoord];
    z_out = new float[numcoord];
    if (DataType == DATA_S)
    {
        u_out = new float[numcoord];
    }
    else if (DataType == DATA_V)
    {
        u_out = new float[numcoord];
        v_out = new float[numcoord];
        w_out = new float[numcoord];
    }
    else if (DataType == DATA_S_E)
    {
        // size of array: assume worst case (6 surfaces per finite element)
        u_out = new float[numelem * 6];
    }
    else if (DataType == DATA_V_E)
    {
        u_out = new float[numelem * 6];
        v_out = new float[numelem * 6];
        w_out = new float[numelem * 6];
    }

    int cl_alloc, ct_alloc, el_alloc;
    if (MEMORY_OPTIMIZED == 1)
    {
        ct_alloc = numcoord;
        // this will double computation-time but minimize memory-usage
        cl_alloc = el_alloc = 0;
        for (i = 0; i < numelem; i++)
            switch (tl[i])
            {
            case TYPE_HEXAGON:
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 4]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7], cl[el[i] + 6]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6], cl[el[i] + 7]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 3]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 5]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                break;
            case TYPE_TETRAHEDER:
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 2], cl[el[i] + 1]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 3]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                break;
            case TYPE_PRISM:
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 3]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 1]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 1]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                break;
            case TYPE_PYRAMID:
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 4], cl[el[i] + 3]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 4]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 3;
                }
                if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]) < 0)
                {
                    el_alloc++;
                    cl_alloc += 4;
                }
                break;
            case TYPE_QUAD:
                el_alloc++;
                cl_alloc += 4;
                break;
            case TYPE_TRIANGLE:
                el_alloc++;
                cl_alloc += 3;
                break;
            }
        // that's it
    }
    else
    {
        cl_alloc = 4 * 6 * numelem;
        ct_alloc = numcoord;
        el_alloc = 6 * numelem;
    }

    conn_list = new int[cl_alloc];
    conn_tag = new int[ct_alloc];
    elem_list = new int[el_alloc];

    memset(conn_tag, -1, numcoord * sizeof(int));

    num_vert = 0;
    num_conn = 0;
    num_elem = 0;
    num_bar = 0;

    int first = 1;
    for (i = 0; i < numelem; i++)
    {
        // compute volume-center of current element
        switch (tl[i])
        {
        case TYPE_HEXAGON:
            c = 8;
            break;
        case TYPE_TETRAHEDER:
            c = 4;
            break;
        case TYPE_PRISM:
            c = 6;
            break;
        case TYPE_PYRAMID:
            c = 5;
            break;
        default:
            // other possible elements are 2D and so we can't compute a
            // volume-center nor can we decide where the normal
            // has to point to
            c = 0;
            break;
        }
        x_center = 0;
        y_center = 0;
        z_center = 0;
        for (a = 0; a < c; a++)
        {
            x_center += x_in[cl[el[i] + a]];
            y_center += y_in[cl[el[i] + a]];
            z_center += z_in[cl[el[i] + a]];
        }
        x_center /= (float)c;
        y_center /= (float)c;
        z_center /= (float)c;

        //converting into polygons
        switch (tl[i])
        {
        case TYPE_HEXAGON:
        {
            //Computation for hexahedra
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 4]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 5]) || test(cl[el[i]], cl[el[i] + 4], cl[el[i] + 5]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 2]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7], cl[el[i] + 6]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7]) || test(cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 7]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 7], cl[el[i]]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 6]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 7]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 6]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 7]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6], cl[el[i] + 7]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6]) || test(cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 6]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 4], cl[el[i] + 5], cl[el[i] + 6], cl[el[i] + 1]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 7]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 6]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 7]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 6]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 3]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 4], cl[el[i] + 7]) || test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 7]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 4], cl[el[i] + 7], cl[el[i] + 5]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 7]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 7]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 5]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6]) || test(cl[el[i] + 1], cl[el[i] + 5], cl[el[i] + 6]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 6], cl[el[i] + 3]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 6]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 6]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) || test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 7]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
        }
        break;

        case TYPE_TETRAHEDER:
        {
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 2], cl[el[i] + 1]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 2], cl[el[i] + 1]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 2], cl[el[i] + 1], cl[el[i] + 3]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 3]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 3], cl[el[i] + 2]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2]) < 0)
            {
                if (test(cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 3], cl[el[i] + 1], cl[el[i] + 2], cl[el[i]]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
        }
        break;

        case TYPE_PRISM:
        {
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 3]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 2], cl[el[i] + 5]) || test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 5]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 1]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]) < 0)
            {
                if (test(cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3], cl[el[i] + 1]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 4]) || test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 5]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4]) || test(cl[el[i] + 2], cl[el[i] + 1], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 2], cl[el[i] + 5], cl[el[i] + 4], cl[el[i] + 3]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 5]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                }
            }
        }
        break;

        case TYPE_PYRAMID:
        {
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 1], cl[el[i] + 4], cl[el[i] + 2]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 4], cl[el[i] + 3]) < 0)
            {
                if (test(cl[el[i]], cl[el[i] + 4], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 4], cl[el[i] + 3], cl[el[i] + 2]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 4]) < 0)
            {
                if (test(cl[el[i] + 4], cl[el[i] + 2], cl[el[i] + 3]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 2], cl[el[i] + 3], cl[el[i] + 4], cl[el[i] + 0]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]) < 0)
            {
                if (test(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i] + 1], cl[el[i] + 2], cl[el[i] + 4], cl[el[i] + 3]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 4]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                    }
                }
            }
            if (tmp_grid->getNeighbor(i, cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 1]) < 0)
            {
                //sc: when two points of a quad are identical
                if (test(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2]) || test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
                {
                    if (DataType == DATA_S_E)
                    {
                        u_out[num_elem] = u_in[i];
                    }
                    else if (DataType == DATA_V_E)
                    {
                        u_out[num_elem] = u_in[i];
                        v_out[num_elem] = v_in[i];
                        w_out[num_elem] = w_in[i];
                    }
                    elem_list[num_elem] = num_conn;
                    num_elem++;
                    if (norm_check(cl[el[i]], cl[el[i] + 3], cl[el[i] + 2], cl[el[i] + 4]))
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                    }
                    else
                    {
                        conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                        num_conn++;
                        conn_list[num_conn] = add_vertex(cl[el[i]]);
                        num_conn++;
                    }
                }
            }
        }
        break;

        case TYPE_QUAD:
        {

            if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
            {
                if (DataType == DATA_S_E)
                {
                    u_out[num_elem] = u_in[i];
                }
                else if (DataType == DATA_V_E)
                {
                    u_out[num_elem] = u_in[i];
                    v_out[num_elem] = v_in[i];
                    w_out[num_elem] = w_in[i];
                }
                elem_list[num_elem] = num_conn;
                num_elem++;
                conn_list[num_conn] = add_vertex(cl[el[i]]);
                num_conn++;
                conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                num_conn++;
                conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                num_conn++;
                conn_list[num_conn] = add_vertex(cl[el[i] + 3]);
                num_conn++;
            }
        }
        break;

        case TYPE_TRIANGLE:
        {

            if (test(cl[el[i]], cl[el[i] + 1], cl[el[i] + 2]))
            {
                if (DataType == DATA_S_E)
                {
                    u_out[num_elem] = u_in[i];
                }
                else if (DataType == DATA_V_E)
                {
                    u_out[num_elem] = u_in[i];
                    v_out[num_elem] = v_in[i];
                    w_out[num_elem] = w_in[i];
                }
                elem_list[num_elem] = num_conn;
                num_elem++;
                conn_list[num_conn] = add_vertex(cl[el[i]]);
                num_conn++;
                conn_list[num_conn] = add_vertex(cl[el[i] + 1]);
                num_conn++;
                conn_list[num_conn] = add_vertex(cl[el[i] + 2]);
                num_conn++;
            }
        }
        break;

        case TYPE_BAR: // no surface representation possible
            num_bar++;
            break;
        case TYPE_POINT: // no surface/line representation possible
            break; // but do not send an error!!!!

        default:
        {
            if (first)
                Covise::sendError("ERROR: unsupported grid type detected");
            first = 0;
            //return;
        }
            //  break; Everything is either specific or default...
        }
    }

    elemMap = new int[num_bar];
    nb = 0;
    for (i = 0; i < numelem; i++)
    {

        switch (tl[i])
        {
        case TYPE_BAR:
            elemMap[nb] = i;
            nb++;
            break;
        };
    }

    return;
}

void Application::paramCallback(bool /*inMapLoading*/, void *userData, void *callbackData)
{
    (void)callbackData;

    Application *thisApp = (Application *)userData;
    const char *paramname = Covise::get_reply_param_name();

    // title of module has changed
    if (0 == strcmp(paramname, "SetModuleTitle"))
    {
        const char *title;
        Covise::get_reply_string(&title);
        thisApp->setTitle(title);
    }
}

void Application::setTitle(const char *title)
{
    delete[] d_title;
    d_title = strcpy(new char[strlen(title) + 1], title);
}

void Application::compute(void *)
{

    cuc = NULL;
    cuc_pos = NULL;

    // ========================== Get mesh ======================

    char *objName;
    coDistributedObject *tmp_obj = NULL;
    coDistributedObject *meshIn = NULL;
    coDistributedObject *dataIn = NULL;

    tresh = (float)0.001;
    Covise::get_scalar_param("angle", &tresh);
    Covise::get_vector_param("vertex", 0, &n2x);
    Covise::get_vector_param("vertex", 1, &n2y);
    Covise::get_vector_param("vertex", 2, &n2z);
    Covise::get_scalar_param("scalar", &angle);
    Covise::get_boolean_param("double", &doDoubleCheck);
    Covise::get_choice_param("optimize", &MEMORY_OPTIMIZED);
    MEMORY_OPTIMIZED--;

    //////// Input Mesh : required

    objName = Covise::get_object_name("meshIn");
    if (objName == NULL)
    {
        Covise::sendError("Error creating object name for 'meshIn'");
        return;
    }
    tmp_obj = new coDistributedObject(objName);
    if ((tmp_obj == NULL) || (!tmp_obj->objectOk()))
    {
        Covise::sendError("Error reading mesh");
        return;
    }
    meshIn = tmp_obj->createUnknown();
    if (meshIn == NULL)
    {
        Covise::sendError("createUnknown() failed for mesh");
        return;
    }
    delete tmp_obj;

    //////// Input Data : not required

    objName = Covise::get_object_name("dataIn");
    if (objName)
    {
        tmp_obj = new coDistributedObject(objName);
        if ((tmp_obj == NULL) || (!tmp_obj->objectOk()))
        {
            Covise::sendError("Error reading input data");
            return;
        }
        dataIn = tmp_obj->createUnknown();
        if (dataIn == NULL)
        {
            Covise::sendError("createUnknown() failed for input data");
            return;
        }
        delete tmp_obj;
    }
    else
        dataIn = NULL;

    coDistributedObject *meshOut, *dataOut, *linesOut, *ldataOut;
    char *meshOutName, *dataOutName, *linesOutName, *ldataOutName;

    //////// Output Data names: Mesh

    meshOutName = Covise::get_object_name("meshOut");
    if (meshOutName == NULL)
    {
        Covise::sendError("Error creating object name for 'meshOut'");
        return;
    }

    //////// Output Data names: Lines

    linesOutName = Covise::get_object_name("linesOut");
    if (linesOutName == NULL)
    {
        Covise::sendError("Error creating object name for 'linesOut'");
        return;
    }

    //////// Output Data names: Data

    if (dataIn)
    {
        dataOutName = Covise::get_object_name("dataOut");
        if (dataOutName == NULL)
        {
            Covise::sendError("Error creating object name for 'dataOut'");
            return;
        }
        ldataOutName = Covise::get_object_name("ldataOut");
        if (ldataOutName == NULL)
        {
            Covise::sendError("Error creating object name for 'ldataOut'");
            return;
        }
    }
    else
    {
        dataOutName = NULL;
        dataOut = NULL;
        ldataOutName = NULL;
        ldataOut = NULL;
    }

    doModule(meshIn, dataIn,
             meshOutName, dataOutName, linesOutName, ldataOutName,
             &meshOut, &dataOut, &linesOut, &ldataOut, 1);

    if (NULL != meshOut)
    {
        meshOut->addAttribute("OBJECTNAME", this->getTitle());
    }
    if (NULL != dataOut)
    {
        dataOut->addAttribute("OBJECTNAME", this->getTitle());
    }
    if (NULL != linesOut)
    {
        linesOut->addAttribute("OBJECTNAME", this->getTitle());
    }
    if (NULL != ldataOut)
    {
        ldataOut->addAttribute("OBJECTNAME", this->getTitle());
    }

    delete meshIn;
    delete dataIn;
    delete meshOut;
    delete dataOut;
    delete linesOut;
    delete ldataOut;
}

////// normals

int Application::norm_check(int v1, int v2, int v3, int v4)
{
    int r;
    float a[3], b[3], c[3], n[3], x;

    return (1);

    // compute normal of a=v2v1 and b=v2v3
    a[0] = x_in[v1] - x_in[v2];
    a[1] = y_in[v1] - y_in[v2];
    a[2] = z_in[v1] - z_in[v2];
    b[0] = x_in[v3] - x_in[v2];
    b[1] = y_in[v3] - y_in[v2];
    b[2] = z_in[v3] - z_in[v2];

    n[0] = a[1] * b[2] - b[1] * a[2];
    n[1] = a[2] * b[0] - b[2] * a[0];
    n[2] = a[0] * b[1] - b[0] * a[1];

    if (v4 == -1)
    {
        // compute vector from base-point to volume-center
        c[0] = x_center - x_in[v2];
        c[1] = y_center - y_in[v2];
        c[2] = z_center - z_in[v2];

        // look if normal is correct or not
        if ((c[0] * n[0] + c[1] * n[1] + c[2] * n[2]) > 0)
            r = 0;
        else
            r = 1;
    }
    else
    {
        // compute vector v2v4
        c[0] = x_in[v4] - x_in[v2];
        c[1] = y_in[v4] - y_in[v2];
        c[2] = z_in[v4] - z_in[v2];

        x = (c[0] * n[0] + c[1] * n[1] + c[2] * n[2]);

        // look if normal is correct or not
        if (x > 0)
            r = 0;
        else if (x < 0)
            r = 1;
        else
            r = norm_check(v1, v2, v3);
    }

    // return wether the orientation is correct (!0) or not (0)

    return (r);
}
