/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:  COVISE Calc application module                           **
 **                                                                        **
 **                                                                        **
 **                             (C) 1996                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Robert Stetter                                                **
 **          Christof Schwenzer                                            **
 **                                                                        **
 ** Date:  21.11.96  V1.0  (Final Version)                                 **
 **        8.12.2000 V2.0                                                  **
\**************************************************************************/

#include "Calc.h"
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

using namespace covise;

//#define DEBUGMODE 1

enum Types
{
    TOKEN,
    SKALAR,
    VEKTOR
};

//  Set legal operations
int pLegal[(MAX_OPERATIONS + MAX_FUNKT) * 12] =
    //  plus  minus    mal   get.   hoch  wurz.  v.pr.    sin    cos    tan    log    exp    neg   vlen  comp1  comp2  comp3    max    min   atan

    { //VEK-VEK
      true, true, true, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
      VEKTOR, VEKTOR, SKALAR, false, false, false, VEKTOR, false, false, false, false, false, false, false, false, false, false, false, false, false,

      //VEK-SKA
      false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
      false, false, VEKTOR, VEKTOR, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,

      //SKA-VEK
      false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
      false, false, VEKTOR, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,

      //SKA-SKA
      true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
      SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, false, false, false, false, false, false, false, false, false, false, false, false, false, false,

      //VEK
      false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, false,
      false, false, false, false, false, false, false, false, false, false, false, false, VEKTOR, SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, false,

      //SKA
      false, false, false, false, false, false, false, true, true, true, true, true, true, true, false, false, false, true, true, true,
      false, false, false, false, false, false, false, SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, SKALAR, false, false, false, SKALAR, SKALAR, SKALAR
    };

//  Local data
int i_dim = 0, j_dim = 0, k_dim = 0, dim = 0; //  Dim. of input object
int i_dim_s1 = 0, j_dim_s1 = 0, k_dim_s1 = 0, dim_s1 = 0; //  Dim. of s1 input object
int i_dim_s2 = 0, j_dim_s2 = 0, k_dim_s2 = 0, dim_s2 = 0; //  Dim. of s2 input object
int i_dim_v1 = 0, j_dim_v1 = 0, k_dim_v1 = 0, dim_v1 = 0; //  Dim. of v1 input object
int i_dim_v2 = 0, j_dim_v2 = 0, k_dim_v2 = 0, dim_v2 = 0; //  Dim. of v2 input object
int str_unstr = 0; //  indicate struct./unstr. data

// input- / output-pointer
float *s1_in, *s2_in, *s_out;
float *u1_in, *v1_in, *w1_in;
float *u2_in, *v2_in, *w2_in;
float *u_out, *v_out, *w_out;

char *dtype_s1, *dtype_s2, *dtype_v1, *dtype_v2;
char *VDataIn[NUMVECTOR], *SDataIn[NUMSCALAR];
const char *VDataOut;
const char *SDataOut;
char *funktion;

//Shared memory data

const coDoFloat *s_data_in_1_unstr = NULL;
const coDoFloat *s_data_in_2_unstr = NULL;
coDoFloat *s_data_out_unstr = NULL;
const coDoVec3 *v_data_in_1_unstr = NULL;
const coDoVec3 *v_data_in_2_unstr = NULL;
coDoVec3 *v_data_out_unstr = NULL;

void
Calc::copyAttributesToOutObj(coInputPort **input_ports,
                             coOutputPort **output_ports,
                             int i)
{
    // how many input objects are there?
    int no_input = 0;
    if (input_ports[0])
    {
        ++no_input;
    }
    if (input_ports[1])
    {
        ++no_input;
    }
    if (input_ports[2])
    {
        ++no_input;
    }
    if (input_ports[3])
    {
        ++no_input;
    }

    //   fprintf(stderr, "no_input=%d, i=%d, connected=%d\n", no_input, i, int(output_ports[i]&&output_ports[i]->getCurrentObject()));

    int j = 0;
    if (output_ports[i] == NULL
        || output_ports[i]->getCurrentObject() == NULL)
    {
        return;
    }
    else if (i == 0) // output is scalar
    {
        if (input_ports[0])
        {
            j = 0;
        }
        else if (input_ports[1])
        {
            j = 1;
        }
        else if (input_ports[2])
        {
            j = 2;
        }
        else if (input_ports[3])
        {
            j = 3;
        }
    }
    else if (i == 1) // output is vector
    {
        if (input_ports[2])
        {
            j = 2;
        }
        else if (input_ports[3])
        {
            j = 3;
        }
        else if (input_ports[0])
        {
            j = 0;
        }
        else if (input_ports[1])
        {
            j = 1;
        }
    }
    else
    {
        return;
    }
    //   fprintf(stderr, "j=%d, input_ports[j]=%p\n", j, input_ports[j]);
    if (input_ports[j])
    {
        //copyAttributes( output_ports[i]->getCurrentObject(), input_ports[j]->getCurrentObject() );
        const char **name, **setting;
        const coDistributedObject *src = input_ports[j]->getCurrentObject();
        coDistributedObject *tgt = output_ports[i]->getCurrentObject();
        if (src && tgt)
        {
            int n = src->getAllAttributes(&name, &setting);
            for (int attr = 0; attr < n; ++attr)
            {
                if (no_input == 1
                    || strcmp(name[attr], "SPECIES") != 0)
                {
                    tgt->addAttribute(name[attr], setting[attr]);
                }
            }
        }
    }
}

Calc::Calc(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Simple calculator module")
{

    p_sInData1 = addInputPort("DataIn0", "Float", "Scalar Data In1");
    p_sInData1->setRequired(false);
    p_sInData2 = addInputPort("DataIn1", "Float", "Scalar Data In2");
    p_sInData2->setRequired(false);
    p_vInData1 = addInputPort("DataIn2", "Vec3|UnstructuredGrid|Polygons", "Vector Data In1");
    p_vInData1->setRequired(false);
    p_vInData2 = addInputPort("DataIn3", "Vec3|UnstructuredGrid|Polygons", "Vector Data In2");
    p_vInData2->setRequired(false);
    p_sOutData1 = addOutputPort("DataOut0", "Float", "Scalar Data Out");
    p_vOutData2 = addOutputPort("DataOut1", "Vec3", "Vector Data Out");
    p_expression = addStringParam("expression", "expression to evaluate");

    setComputeTimesteps(0);
    setComputeMultiblock(0);
}

//
//
//..........................................................................
//
//

// set coSimpleModule::portLeader, the user may leave
// in_ports[0] unused
void
Calc::preHandleObjects(coInputPort **in_ports)
{
    int port;
    for (port = 0; port < 4; ++port)
    {
        if (in_ports[port]->getCurrentObject())
        {
            portLeader = port; // portLeader: see coSimpleModule
            break;
        }
    }
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
int Calc::compute(const char *)
{
    unsigned int i;
    char buf[400];

    // get object names for scalar input data
    for (i = 0; i < NUMSCALAR; i++)
    {
        sprintf(buf, "s_indata%d", i + 1);
        SDataIn[i] = Covise::get_object_name(buf);
    }

    // get object names for vector input data
    for (i = 0; i < NUMVECTOR; i++)
    {
        sprintf(buf, "v_indata%d", i + 1);
        VDataIn[i] = Covise::get_object_name(buf);
    }

    // get current expression
    Covise::get_string_param("expression", &funktion);

#ifdef DEBUGMODE
    strcpy(buf, "new expression to evaluate: ");
    strcat(buf, funktion);
    sendInfo(buf);
#endif

    // start calculation

    CCalc *pCalc;

    pCalc = new CCalc();
    pCalc->Compute(this, funktion);
    delete pCalc;

    //cerr<< "Calc::compute" <<endl;
    return CONTINUE_PIPELINE;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Implementation of class CCalc                                            //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

CCalc::CCalc()
    : pList(NULL)
    , pListPostfix(NULL)
    , head(NULL)
    , end(NULL)
    , stack_item(NULL)
    , head_eval(NULL)
    , end_eval(NULL)
    , item_eval(NULL)
    , pVektor_1(NULL)
    , pVektor_2(NULL)
    , pEinh_Vektor(NULL)
    , pMan_Vektor(NULL)
    , Count_Man_Vektors(0)
    , pSkalar_1(NULL)
    , pSkalar_2(NULL)
    , Array_Len(0)
    , Vek_Len(0)
    , Result_Vektor(NULL)
    , Result_Skalar(0.f)
{
}

CCalc::~CCalc() // destructor
{
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Compute: main modul                                                      //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::Compute(Calc *module, char *Expression)
{
    char *String;
    int TempVektors = 0; // number of intermediate results with type VEKTOR
    int Array_Count = 0; // counter for data-array
    float *pIntermed = NULL; // Pointer to beginning of intermediate results-section
    int Anz_Man_Vekt = 0; // number of manual vectors

    const coDistributedObject *v_obj_1, *v_obj_2, *s_obj_1, *s_obj_2;
    //   coDistributedObject *tmp_obj_s1, *tmp_obj_s2, *tmp_obj_v1, *tmp_obj_v2;
    int str_s1 = 0, str_s2 = 0, str_v1 = 0, str_v2 = 0;

    // setting vector dimension
    //
    // by now only 3-dim vectors supported
    // to add 2-dim vectors modify the following sections:
    //	*retrieve data object from shared memory
    //	*Test for structured/unstructured mismatch
    //	*Test for dimensionen mismatch
    //	*get output data object names and create objects
    // all other functions are able to handle 2-dim data.

    Vek_Len = 3;

    // resetting data object dimensions
    i_dim = 0;
    j_dim = 0;
    k_dim = 0;
    dim = 0;
    i_dim_s1 = 0;
    j_dim_s1 = 0;
    k_dim_s1 = 0;
    dim_s1 = 0;
    i_dim_s2 = 0;
    j_dim_s2 = 0;
    k_dim_s2 = 0;
    dim_s2 = 0;
    i_dim_v1 = 0;
    j_dim_v1 = 0;
    k_dim_v1 = 0;
    dim_v1 = 0;
    i_dim_v2 = 0;
    j_dim_v2 = 0;
    k_dim_v2 = 0;
    dim_v2 = 0;

    // resetting data object pointers
    s1_in = NULL;
    s2_in = NULL;
    u1_in = NULL;
    u2_in = NULL;

    // resetting datatypes

    dtype_s1 = NULL;
    dtype_s2 = NULL;
    dtype_v1 = NULL;
    dtype_v2 = NULL;

    // retrieve data object from shared memory
    //neu
    // scalar dat from input 1
    s_obj_1 = module->p_sInData1->getCurrentObject();
    if (NULL != s_obj_1)
    {
        s_data_in_1_unstr = (coDoFloat *)s_obj_1;
        dim_s1 = s_data_in_1_unstr->getNumPoints();
        s_data_in_1_unstr->getAddress(&s1_in);
        str_s1 = UNSTR;
    }
    // scalar dat from input 2
    s_obj_2 = module->p_sInData2->getCurrentObject();
    if (NULL != s_obj_2)
    {
        s_data_in_2_unstr = (coDoFloat *)s_obj_2;
        dim_s2 = s_data_in_2_unstr->getNumPoints();
        s_data_in_2_unstr->getAddress(&s2_in);
        str_s2 = UNSTR;
    }
    //vector data from ( vector ) input 1
    v_obj_1 = module->p_vInData1->getCurrentObject();
    if (NULL != v_obj_1)
    {
        v_data_in_1_unstr = dynamic_cast<const coDoVec3 *>(v_obj_1);
        if (v_data_in_1_unstr)
        {
            dim_v1 = v_data_in_1_unstr->getNumPoints();
            v_data_in_1_unstr->getAddresses(&u1_in, &v1_in, &w1_in);
            str_v1 = UNSTR;
        }
        else if (const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid *>(v_obj_1))
        {
            int test;
            grid->getGridSize(&test, &test, &dim_v1);
            int *dummy;
            grid->getAddresses(&dummy, &dummy, &u1_in, &v1_in, &w1_in);
            str_v1 = UNSTR;
        }
        else if (const coDoPolygons *grid = dynamic_cast<const coDoPolygons *>(v_obj_1))
        {
            dim_v1 = grid->getNumPoints();
            int *dummy;
            grid->getAddresses(&u1_in, &v1_in, &w1_in, &dummy, &dummy);
            str_v1 = UNSTR;
        }
    }

    //	vector data from input 2
    v_obj_2 = module->p_vInData2->getCurrentObject();
    if (NULL != v_obj_2)
    {
        v_data_in_2_unstr = dynamic_cast<const coDoVec3 *>(v_obj_2);
        if (v_data_in_2_unstr)
        {
            dim_v2 = v_data_in_2_unstr->getNumPoints();
            v_data_in_2_unstr->getAddresses(&u2_in, &v2_in, &w2_in);
            str_v2 = UNSTR;
        }
        else if (const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid *>(v_obj_2))
        {
            int test;
            grid->getGridSize(&test, &test, &dim_v2);
            int *dummy;
            grid->getAddresses(&dummy, &dummy, &u2_in, &v2_in, &w2_in);
            str_v2 = UNSTR;
        }
        else if (const coDoPolygons *grid = dynamic_cast<const coDoPolygons *>(v_obj_2))
        {
            dim_v2 = grid->getNumPoints();
            int *dummy;
            grid->getAddresses(&u2_in, &v2_in, &w2_in, &dummy, &dummy);
            str_v2 = UNSTR;
        }
    }

    //Test for structured/unstructured mismatch
    //Test for dimension mismatch
    if (!CheckInputs(module, str_s1, str_s2, str_v1, str_v2,
                     i_dim_s1, j_dim_s1, k_dim_s1,
                     i_dim_s2, j_dim_s2, k_dim_s2,
                     i_dim_v1, j_dim_v1, k_dim_v1,
                     i_dim_v2, j_dim_v2, k_dim_v2,
                     dim_s1, dim_s2, dim_v1, dim_v2,
                     &Array_Len, &str_unstr,
                     &i_dim, &j_dim, &k_dim, &dim))
    {
        //in case of an error free memory before return
        delete[](dtype_s1);
        delete[](dtype_s2);
        delete[](dtype_v1);
        delete[](dtype_v2);
        return coModule::FAIL;
    }

    minmax = 0; // switch for message: max/min value in scalars/vectors

    pVektor_1 = new float[Vek_Len];
    pVektor_2 = new float[Vek_Len];
    pEinh_Vektor = new float[Vek_Len];

    Expression = LowerCase(Expression);
    strcpy(Ausdruck, Expression);

    Count_Man_Vektors = 0;
    String = new char[strlen(Ausdruck) + 1];
    strcpy(String, Ausdruck);

    //build list with items contained in the expression
    BuildItemList(String);

    //Check syntax / get number of manual vectors
    if (!CheckSyntax(module, &Anz_Man_Vekt))
    {
        //in case of an error free memory before return
        DeleteItemList();
        delete[](pVektor_1);
        delete[](pVektor_2);
        delete[](pEinh_Vektor);
        delete[](String);
        delete[](dtype_s1);
        delete[](dtype_s2);
        delete[](dtype_v1);
        delete[](dtype_v2);
        return coSimpleModule::FAIL;
    }

    //get memory for manual vectors
    pMan_Vektor = new float[Anz_Man_Vekt * Vek_Len];

    //change from infix to postfix / build manual vectors
    if (!InfixToPostfix(module))
    {
        //in case of an error free memory before return
        DeleteItemList();
        DeletePostfixList();
        delete[](pVektor_1);
        delete[](pVektor_2);
        delete[](pEinh_Vektor);
        delete[](pMan_Vektor);
        delete[](String);
        delete[](dtype_s1);
        delete[](dtype_s2);
        delete[](dtype_v1);
        delete[](dtype_v2);
        return coSimpleModule::FAIL;
    }

    //get type of result and size of memory needed for intermediate results
    if (!GetResultType(module, &Result_Type, &TempVektors))
    {
        //in case of an error free memory before return
        DeleteItemList();
        DeletePostfixList();
        delete[](pVektor_1);
        delete[](pVektor_2);
        delete[](pEinh_Vektor);
        delete[](pMan_Vektor);
        delete[](String);
        delete[](dtype_s1);
        delete[](dtype_s2);
        delete[](dtype_v1);
        delete[](dtype_v2);
        return coSimpleModule::FAIL;
    }

    // get memory for intermediate results
    pIntermed = (float *)new float[(TempVektors + 1) * Vek_Len];

    //	get output data object names and create objects
    switch (Result_Type)
    {
    case SKALAR: //structured scalar data
        SDataOut = module->p_sOutData1->getObjName();
#ifdef DEBUGMODE
        cout << "dim: " << dim << endl;
#endif
        s_data_out_unstr = new coDoFloat(SDataOut, dim);
        if (s_data_out_unstr->objectOk())
        {
            s_data_out_unstr->getAddress(&s_out);
            module->p_sOutData1->setCurrentObject(s_data_out_unstr);
        }
        else
        {
            module->sendError("ERROR: creation of data object 'outdata1' failed");
            //module->sendError( SDataOut );
            return coSimpleModule::FAIL;
        }
        break;

    case VEKTOR: //structured vector data
        VDataOut = module->p_vOutData2->getObjName();
        //cerr << "V: " << SDataOut << "  " << str_unstr << endl;
        //cerr << i_dim << "  " << j_dim << "  " << k_dim << endl;
        v_data_out_unstr = new coDoVec3(VDataOut, dim);
        if (v_data_out_unstr->objectOk())
        {
            v_data_out_unstr->getAddresses(&u_out, &v_out, &w_out);
            module->p_vOutData2->setCurrentObject(v_data_out_unstr);
        }
        else
        {
            module->sendError("ERROR: creation of data object 'outdata2' failed");
            return coSimpleModule::FAIL;
        }
        break;
    }

    //evaluate array

    for (Array_Count = 0; Array_Count < Array_Len; Array_Count++)
    {
        Result_Vektor = pIntermed;
        if (u1_in != NULL) // vector-input 1 connected
        {
            pVektor_1[0] = u1_in[Array_Count];
            pVektor_1[1] = v1_in[Array_Count];
            if (Vek_Len == 3)
                pVektor_1[2] = w1_in[Array_Count];
        }

        if (u2_in != NULL) // vector-input 2 connected
        {
            pVektor_2[0] = u2_in[Array_Count];
            pVektor_2[1] = v2_in[Array_Count];
            if (Vek_Len == 3)
                pVektor_2[2] = w2_in[Array_Count];
        }

        if (s1_in != NULL)
            pSkalar_1 = &s1_in[Array_Count];
        if (s2_in != NULL)
            pSkalar_2 = &s2_in[Array_Count];

        //evaluation of expression
        if (!Evaluate(module, &Result_Type, &Result_Vektor, &Result_Skalar))
        {
            //in case of an error free memory before return
            DeleteItemList();
            DeletePostfixList();

            delete[](pVektor_1);
            delete[](pVektor_2);
            delete[](pEinh_Vektor);
            delete[](pMan_Vektor);
            delete[](String);
            delete[](pIntermed);
            delete[](dtype_s1);
            delete[](dtype_s2);
            delete[](dtype_v1);
            delete[](dtype_v2);
            return coSimpleModule::FAIL;
        }

        //write result into output object
        switch (Result_Type)
        {
        case VEKTOR:
            u_out[Array_Count] = Result_Vektor[0];
            v_out[Array_Count] = Result_Vektor[1];
            if (Vek_Len == 3)
                w_out[Array_Count] = Result_Vektor[2];
            break;

        case SKALAR:
            s_out[Array_Count] = Result_Skalar;
            break;
        }
        if (Array_Count == 0)
            minmax = 1; //in case of max/min: send message
    }

    //FOR TESTING *********************************************************
    //write result into file
    //only for testing
    //

    /*FILE *fp;
     fp = Covise::fopen("sgi/bin/test_calc.out", "w");

     fprintf (fp,"Testfile\n\n");
     fprintf (fp,"Ausdruck: %s\n\n", Ausdruck);

     for (Array_Count=0; Array_Count < Array_Len; Array_Count++)
     {
     // read results out of output object
     switch (Result_Type)
   {
   case VEKTOR:  	fprintf(fp,"%f ",u_out[Array_Count]);
   fprintf(fp,"%f ",v_out[Array_Count]);
   if(Vek_Len == 3) fprintf(fp,"%f \n",w_out[Array_Count]);
   break;

   case SKALAR:  fprintf(fp,"%f \n",s_out[Array_Count]);
   break;
   }

   }

   fclose (fp);*/

    //
    //FOR TESTING *********************************************************

    //free memory
    FreeMemory();
    delete[](String);
    delete[](pIntermed);
    delete[](dtype_s1);
    delete[](dtype_s2);
    delete[](dtype_v1);
    delete[](dtype_v2);
    return coSimpleModule::SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// BuildItemList: build list with items contained in the expression         //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void
CCalc::BuildItemList(char *StringExpr)
{
    int Count = 0;
    char *Item;
    char *TempString;
    int Priority, Type, Token;
    int Space = 0;

    pList = new LIST[(strlen(Ausdruck) + 1)];
    TempString = new char[strlen(Ausdruck) + 1];

    //get items out of string
    while (*StringExpr != '\0')
    {
        strcpy(TempString, StringExpr);

        //get one item
        ReadItem(TempString, &Priority, &Type, &Token, &Space);

        if (!Space)
        {
            Item = new char[strlen(Ausdruck) + 1];
            strcpy(Item, TempString);
            pList[Count].Item = Item;
            pList[Count].Priority = Priority;
            pList[Count].Type = Type;
            pList[Count].Token = Token;
            StringExpr = StringExpr + strlen(TempString);
            Count++;
        }

        //ingnore <SPACE> in string
        else
        {
            StringExpr++;
        }
    }
    delete[] TempString;

    //build last entry in list
    Item = new char[5];
    strcpy(Item, "End");
    pList[Count].Item = Item;
    pList[Count].Priority = EOL;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// ReadItem: get one item out of string                                     //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::ReadItem(char *String, int *Priority, int *Type, int *Op,
                     int *Leerz)
{
    char *Temp;
    char Token[2];

    *Leerz = false;

    //find first occurance of one of these token
    Temp = strpbrk(String, " +-*/^()~#[],");
    if (Temp != NULL)
    {
        //write first token found into Token
        Token[0] = Temp[0];
        Token[1] = '\0';
    }
    else
        Token[0] = '\0';

    switch (String[0])
    {
    case '+':
        *Priority = 2;
        *Op = PLUS;
        strcpy(String, "+");
        break;
    case '-':
        *Priority = 2;
        *Op = MINUS;
        strcpy(String, "-");
        break;
    case '*':
        *Priority = 3;
        *Op = MAL;
        strcpy(String, "*");
        break;
    case '/':
        *Priority = 3;
        *Op = GETEILT;
        strcpy(String, "/");
        break;
    case '^':
        *Priority = 4;
        *Op = HOCH;
        strcpy(String, "^");
        break;
    case '~':
        *Priority = 4;
        *Op = WURZEL;
        strcpy(String, "~");
        break;
    case '#':
        *Priority = 3;
        *Op = VEK_PROD;
        strcpy(String, "#");
        break;
    case '(':
        *Priority = 1;
        *Op = NO_OP;
        strcpy(String, "(");
        break;
    case ')':
        *Priority = 6;
        *Op = NO_OP;
        strcpy(String, ")");
        break;
    case '[':
        *Priority = 7;
        *Op = VEK_START;
        strcpy(String, "[");
        break;

    case ']':
        *Priority = 8;
        *Op = VEK_END;
        strcpy(String, "]");
        break;

    case ',':

    case ' ':
        *Leerz = true;
        break;

    default: // operand or function
        strcpy(String, strtok(String, Token));
        *Op = 0;
        if (!strcmp(String, SINUS))
            *Op = SIN;
        if (!strcmp(String, COSINUS))
            *Op = COS;
        if (!strcmp(String, TANGENS))
            *Op = TAN;
        if (!strcmp(String, ARCTAN))
            *Op = ATAN;
        if (!strcmp(String, LOGARITH))
            *Op = LOG;
        if (!strcmp(String, EXPON))
            *Op = EXP;
        if (!strcmp(String, NEGATIV))
            *Op = NEG;
        if (!strcmp(String, VEKTOR_LEN))
            *Op = VLEN;
        if (!strcmp(String, COMPOUND_1))
            *Op = COMP_1;
        if (!strcmp(String, COMPOUND_2))
            *Op = COMP_2;
        if (!strcmp(String, COMPOUND_3))
            *Op = COMP_3;
        if (!strcmp(String, MAXIMUM))
            *Op = MAX;
        if (!strcmp(String, MINIMUM))
            *Op = MIN;
        if (*Op)
            *Priority = 5; //funtion
        else
        {
            *Priority = 0; //operand
            *Op = OPERAND;
        }
    }
    if (*Op == OPERAND)
        *Type = GetType(String); //get type of operand (scalar/vector)
    else
        *Type = TOKEN;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// LowerCase:  change string to lower cases                                 //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

char *CCalc::LowerCase(char *string)
{
    char *cp;

    for (cp = string; *cp; ++cp)
    {
        if ('A' <= *cp && *cp <= 'Z')
            *cp += 'a' - 'A';
    }
    return (string);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Maximum:  get maximum out of four int-values                             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::Maximum(int Zahl1, int Zahl2, int Zahl3, int Zahl4)
{
    int puffer;

    puffer = Zahl1;

    if (Zahl2 > puffer)
        puffer = Zahl2;
    if (Zahl3 > puffer)
        puffer = Zahl3;
    if (Zahl4 > puffer)
        puffer = Zahl4;

    return (puffer);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Minimum:  get minimum out of four int-values                             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::Minimum(int Zahl1, int Zahl2, int Zahl3, int Zahl4)
{
    int puffer;

    puffer = Maximum(Zahl1, Zahl2, Zahl3, Zahl4);

    if (Zahl1 < puffer && Zahl1 != 0)
        puffer = Zahl1;
    if (Zahl2 < puffer && Zahl2 != 0)
        puffer = Zahl2;
    if (Zahl3 < puffer && Zahl3 != 0)
        puffer = Zahl3;
    if (Zahl4 < puffer && Zahl4 != 0)
        puffer = Zahl4;

    return (puffer);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// FreeMemory:  free memory (as the name says)                              //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::FreeMemory()
{
    DeleteItemList(); //free memory needed for item-list
    DeletePostfixList(); //free memory needed for postfix-list

    delete[](pVektor_1);
    delete[](pVektor_2);
    delete[](pEinh_Vektor);
    delete[](pMan_Vektor);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// DeleteItemList:  free memory needed for item-list                        //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::DeleteItemList()
{
    int x = 0;

    while (pList[x].Priority != EOL)
    {
        delete[](pList[x].Item); //delete item
        x++;
    }
    delete[](pList[x].Item); //delete last item
    delete[] pList; //delete list
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// DeletePostfixList:  free memory needed for postfix-list                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::DeletePostfixList()
{
    int x = 0;

    while (pListPostfix[x].Priority != EOL)
    {
        delete[](pListPostfix[x].Item); //delete item
        x++;
    }
    delete[](pListPostfix[x].Item); //delete last item
    delete[] pListPostfix; //delete list
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// CheckSyntax:  check expression for syntax                                //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::CheckSyntax(Calc *module, int *AnzManVek)
{
    bool Done = false;
    int Count = 0;
    int Count_Skalar = 0;
    int Count_Vektor = 0;
    int Count_Operand = 0;
    int Count_Operator = 0;
    int Count_Funktion = 0;
    int Kl_Auf = 0;
    int Kl_Zu = 0;
    int Eck_Kl_Auf = 0;
    int Eck_Kl_Zu = 0;
    bool OK = true;
    unsigned int x = 0;

    while (Done != true)
    {
        OK = true;
        switch (pList[Count].Priority)
        {
        case 0: // operand
            Count_Operand++;
            if (!strcmp(pList[Count].Item, VEKTOR_1))
            {
                Count++;
                Count_Vektor++;
                if (module->p_vInData1->getCurrentObject() == NULL)
                {
                    OK = false;
                    module->sendError("v1 in expression, but not connected");
                }
                continue;
            }
            if (!strcmp(pList[Count].Item, VEKTOR_2))
            {
                Count++;
                Count_Vektor++;
                if (module->p_vInData2->getCurrentObject() == NULL)
                {
                    OK = false;
                    module->sendError("v2 in expression but not connected");
                }
                continue;
            }
            if (!strcmp(pList[Count].Item, SKALAR_1))
            {
                Count++;
                Count_Skalar++;
                if (module->p_sInData1->getCurrentObject() == NULL)
                {
                    OK = false;
                    module->sendError("s1 in expression but not connected");
                }
                continue;
            }
            if (!strcmp(pList[Count].Item, SKALAR_2))
            {
                Count++;
                Count_Skalar++;
                if (module->p_sInData2->getCurrentObject() == NULL)
                {
                    OK = false;
                    module->sendError("s2 in expression but  connected");
                }
                continue;
            }
            if (!strcmp(pList[Count].Item, EINHEITS_V))
            {
                Count++;
                continue;
            }

            //is operand a digit?
            for (x = 0; x < (strlen(pList[Count].Item)); x++)
            {
                if (!isdigit(pList[Count].Item[x])) //no digit
                {
                    //no float expression?
                    if ((pList[Count].Item)[x] != '.')
                    {
                        OK = false; //syntax-error
                        x = (int)strlen(pList[Count].Item);
                    }
                }
            }
            if (!OK)
                Done = true;
            break;

        case 1: // opening parenthesis
            Kl_Auf++;
            break;

        case 2: // operator

        case 3: // operator

        case 4: // operator
            Count_Operator++;
            break;

        case 5: // operator
            Count_Funktion++;
            break;

        case 6: // closing parenthesis
            Kl_Zu++;
            break;

        case 7: // Eckige Klammer auf
            Eck_Kl_Auf++;
            break;

        case 8: // Eckige Klammer zu
            Eck_Kl_Zu++;
            break;

        case EOL: // end of list
            Done = true;
            break;
        }
        Count++;
    }
    if (!OK) //syntax error
    {
        module->sendError("ERROR: syntax error in expression");
        return (0);
    }

    if (Kl_Auf < Kl_Zu) //missing '('
    {
        module->sendError("ERROR: missing '(' in expression");
        return (0);
    }
    if (Kl_Auf > Kl_Zu) //missing ')'
    {
        module->sendError("ERROR: missing ')' in expression");
        return (0);
    }
    if (Eck_Kl_Auf < Eck_Kl_Zu) //missing '['
    {
        module->sendError("ERROR: missing '[' in expression");
        return (0);
    }
    if (Eck_Kl_Auf > Eck_Kl_Zu) //missing ']'
    {
        module->sendError("ERROR: missing ']' in expression");
        return (0);
    }
    if (Count < 4) //too few arguments
    {
        module->sendError("ERROR: too few arguments in expression");
        return (0);
    }

    //get right number of operands if manual vectors are used
    Count_Operand = Count_Operand - (Eck_Kl_Auf * Vek_Len - Eck_Kl_Auf);

    //too few operands
    if (Count_Operator != Count_Operand - 1 || Count_Operator + Count_Funktion == 0)
    {
        module->sendError("ERROR: too few operands in expression");
        return (0);
    }

    // no scalar or vector in expression
    if (Count_Vektor == 0 && Count_Skalar == 0)
    {
        module->sendError("WARNING: expression contains no scalar or vector input");
        dim = 1;
        Array_Len = 1;
        str_unstr = UNSTR;
    }

    *AnzManVek = Eck_Kl_Auf;
    return (1);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// GetType:  get type of operand (scalar/vector)                            //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::GetType(char *Item)
{
    int Op_Type;

    switch (Item[0])
    {
    case 'e': //vector [1,1,1]

    case 'v': //vector input
        Op_Type = VEKTOR;
        break;
    case 's': //scalar input
        Op_Type = SKALAR;
        break;
    default: //manual scalar
        Op_Type = SKALAR;
        break;
    }
    return (Op_Type);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Stackinit: initialize stack  (stack for InfixToPostfix)                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::Stackinit()
{
    head = new /*struct*/ NODE;
    end = new /*struct*/ NODE;
    head->next = end;
    strcpy(head->Item, "\0");
    head->Operator = 0;
    head->Priority = 0;
    end->next = end;
    end->Priority = 0;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Push: push item onto stack   (stack for InfixToPostfix)                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::Push(char *Item, int Operator, int Priority)
{
    stack_item = new /*struct*/ NODE;
    strcpy(stack_item->Item, Item);
    stack_item->Operator = Operator;
    stack_item->Priority = Priority;
    stack_item->next = head->next;
    head->next = stack_item;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Pop: pop item from stack        (stack for InfixToPostfix)               //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::Pop(char *PopItem, int *Operator)
{
    int Priority;

    stack_item = head->next;
    head->next = stack_item->next;
    strcpy(PopItem, stack_item->Item);
    *Operator = stack_item->Operator;
    Priority = stack_item->Priority;
    delete (stack_item);
    return (Priority);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Stack_Free: free stack (what else?)                                      //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::Stack_Free()
{
    delete (head);
    delete (end);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Stackinit: initialize stack (stack for Evaluate)                         //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::Stackinit_Eval()
{
    head_eval = new /*struct*/ NODE_EVAL;
    end_eval = new /*struct*/ NODE_EVAL;

    head_eval->next = end_eval;
    head_eval->Skalar = 0;
    head_eval->pVektor = NULL;
    head_eval->V_Len = 0;
    head_eval->Type = 0;

    end_eval->next = end_eval;
    end_eval->Skalar = 0;
    end_eval->pVektor = NULL;
    end_eval->V_Len = 0;
    end_eval->Type = 0;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Push: push item onto stack  (stack for Evaluate)                         //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::Push(float Item, int Type, float *pVektor, int V_Len)
{
    item_eval = new /*struct*/ NODE_EVAL;

    item_eval->Skalar = Item;
    item_eval->pVektor = pVektor;
    item_eval->V_Len = V_Len;
    item_eval->Type = Type;
    item_eval->next = head_eval->next;

    head_eval->next = item_eval;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Pop: pop item from stack       (stack for Evaluate)                      //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

float CCalc::Pop(int *Type, float **pVektor, int *V_Len)
{
    float PopSkalar;

    item_eval = head_eval->next;
    head_eval->next = item_eval->next;

    PopSkalar = item_eval->Skalar;
    *Type = item_eval->Type;
    *V_Len = item_eval->V_Len;
    *pVektor = item_eval->pVektor;

    delete (item_eval);
    return PopSkalar;
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Stack_Eval_Free: free stack       (Stack für Evaluate)                   //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

void CCalc::Stack_Eval_Free()
{
    delete (head_eval);
    delete (end_eval);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// InfixToPostfix: change expression from infix to postfix                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::InfixToPostfix(Calc *module)
{
    int Count = 0;
    int CountPostfix = 0;
    char *Temp;
    char *Item;
    int Op;
    bool Done = false;
    int y;
    char buffer[MAXLEN];

    pListPostfix = new LIST[(strlen(Ausdruck) + 1)];
    Stackinit();

    while (Done != true)
    {
        switch (pList[Count].Priority)
        {
        case 0: // operand: append to list
            Temp = new char[1 + strlen(Ausdruck)];
            strcpy(Temp, pList[Count].Item);
            pListPostfix[CountPostfix].Item = Temp;
            pListPostfix[CountPostfix].Priority = pList[Count].Priority;
            pListPostfix[CountPostfix].Type = pList[Count].Type;
            pListPostfix[CountPostfix].Token = pList[Count].Token;
            CountPostfix++;
            break;

        case 1: // opening parenthesis: push onto stack
            Push(pList[Count].Item, pList[Count].Token, pList[Count].Priority);
            break;

        case 2: //operator

        case 3: //operator

        case 4: //operator

        case 5: //operator

            if (pList[Count].Priority <= head->next->Priority)
            {
                //pop from stack until priority in infix-list > priority on stack
                while (pList[Count].Priority <= head->next->Priority
                       && head->next != end)
                {
                    Temp = new char[1 + strlen(Ausdruck)];
                    pListPostfix[CountPostfix].Priority = Pop(Temp, &Op);
                    pListPostfix[CountPostfix].Item = Temp;
                    pListPostfix[CountPostfix].Type = TOKEN;
                    pListPostfix[CountPostfix].Token = Op;
                    CountPostfix++;
                }
            }
            Push(pList[Count].Item, pList[Count].Token, pList[Count].Priority);
            break;

        case 6: // closing parenthesis
            while (head->next->Priority != 1)
            { // pop from stack until a "(" or a function is reached
                Temp = new char[1 + strlen(Ausdruck)];
                pListPostfix[CountPostfix].Priority = Pop(Temp, &Op);
                pListPostfix[CountPostfix].Item = Temp;
                pListPostfix[CountPostfix].Type = TOKEN;
                pListPostfix[CountPostfix].Token = Op;
                CountPostfix++;
            }
            if (head->next->Priority == 1)
            { // remove opening parenthsis from stack
                stack_item = head->next;
                head->next = stack_item->next;
                delete (stack_item);
                if (head->next->Priority == 5) //function in front of "("
                {
                    Temp = new char[1 + strlen(Ausdruck)];
                    pListPostfix[CountPostfix].Priority = Pop(Temp, &Op);
                    pListPostfix[CountPostfix].Item = Temp;
                    pListPostfix[CountPostfix].Type = TOKEN;
                    pListPostfix[CountPostfix].Token = Op;
                    CountPostfix++;
                }
            }
            break;

        case 7: // Eckige Klammer auf: manual vector (beginning)
            Count++;
            y = 0;
            while (pList[Count].Priority != 8)
            { //read items out of list until "]" is reached
                if (y >= Vek_Len)
                { //too many elements
                    module->sendError("ERROR: too many elements in manual vector");
                    Item = new char[MAXLEN];
                    strcpy(Item, "End");
                    pListPostfix[CountPostfix].Item = Item;
                    pListPostfix[CountPostfix].Priority = EOL;
                    Stack_Free();
                    return (0);
                }
                if (pList[Count].Type != SKALAR)
                { //non scalar in manual vector
                    module->sendError("ERROR: non scalar element in manual vector");
                    Item = new char[MAXLEN];
                    strcpy(Item, "End");
                    pListPostfix[CountPostfix].Item = Item;
                    pListPostfix[CountPostfix].Priority = EOL;
                    Stack_Free();
                    return (0);
                }
                if (pList[Count].Item[0] == 's')
                { //only digitis are legal
                    module->sendError("ERROR: elements from scalar input not allowed in manual vector");
                    Item = new char[MAXLEN];
                    strcpy(Item, "End");
                    pListPostfix[CountPostfix].Item = Item;
                    pListPostfix[CountPostfix].Priority = EOL;
                    Stack_Free();
                    return (0);
                }
                pMan_Vektor[Count_Man_Vektors * Vek_Len + y] = (float)atof(pList[Count].Item);
                y++;
                Count++;
            }
            Count_Man_Vektors++;
            pListPostfix[CountPostfix].Priority = 0;
            Temp = new char[1 + strlen(Ausdruck)];
            strcpy(Temp, "man"); //write man<number> into list

            sprintf(buffer, "%d", (Count_Man_Vektors - 1));

            strcat(Temp, buffer);
            pListPostfix[CountPostfix].Item = Temp;
            pListPostfix[CountPostfix].Type = VEKTOR;
            pListPostfix[CountPostfix].Token = OPERAND;
            CountPostfix++;
            break;

        case 8: // Eckige Klammer zu

            break;

        case EOL: // end of list: empty stack
            while (head->next != end)
            {
                Temp = new char[1 + strlen(Ausdruck)];
                pListPostfix[CountPostfix].Priority = Pop(Temp, &Op);
                pListPostfix[CountPostfix].Item = Temp;
                pListPostfix[CountPostfix].Type = TOKEN;
                pListPostfix[CountPostfix].Token = Op;
                CountPostfix++;
            }
            Done = true;
            break;
        }
        Count++;
    }

    //build end of postfix-list
    Item = new char[MAXLEN];
    strcpy(Item, "End");
    pListPostfix[CountPostfix].Item = Item;
    pListPostfix[CountPostfix].Priority = EOL;
    Stack_Free();
    return (1);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// GetResultType: get type of result                                        //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::GetResultType(Calc *module, int *Result_Type, int *TempVek)
{
    int CountPostfix = 0;
    bool Done = false;
    int Op_Type_1 = 0, Op_Type_2 = 0;
    int Legal = 0;
    int Operation = 0;
    float *TempVektor_1 = NULL;
    float *TempVektor_2 = NULL;
    int Temp_V_Len_1 = 0;
    int Temp_V_Len_2 = 0;
    int Type_Res = TOKEN;
    int CountTempVektors = 0;

    Stackinit_Eval();

    while (Done != true)
    {
        switch (pListPostfix[CountPostfix].Priority)
        {
        case 0: // operand: push onto stack
            //scalar
            if (pListPostfix[CountPostfix].Type == SKALAR)
            {
                Push(0, SKALAR, NULL, 0);
                CountPostfix++;
                break;
            }

            //vector
            if (pListPostfix[CountPostfix].Type == VEKTOR)
            {
                Push(0, VEKTOR, NULL, 0);
                CountPostfix++;
                break;
            }

        case 2: // operator
        case 3:
        case 4:
        case 5:
            //no function
            if (pListPostfix[CountPostfix].Priority != 5)
                Pop(&Op_Type_2, &TempVektor_2, &Temp_V_Len_2);

            else
                Op_Type_2 = false;

            Pop(&Op_Type_1, &TempVektor_1, &Temp_V_Len_1);
            Legal = CheckOp(Op_Type_1, Op_Type_2,
                            pListPostfix[CountPostfix].Token, &Operation, &Type_Res);
            if (!Legal)
            {
                module->sendError("ERROR: non legal operation in expression");
                return (0);
            }
            if (Type_Res == VEKTOR)
                CountTempVektors++;
            Push(0, Type_Res, NULL, 0);
            CountPostfix++;
            break;

        case EOL: // end of list: type of result is on stack
            *Result_Type = Type_Res;
            delete (item_eval);
            Done = true;
            break;
        }
    }

    Stack_Eval_Free();
    *TempVek = CountTempVektors;
    return (1);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Evaluate: evaluate postfix-expression                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::Evaluate(Calc *module, int *Result_Type, float **Vektor_Res,
                    float *Result_Skalar)
{
    int CountPostfix = 0;
    bool Done = false;
    int Op_Type_1 = 0, Op_Type_2 = 0;
    int Legal = 0;
    int Operation = 0;
    float Temp = 0.0f;
    float *TempVektor_1 = NULL;
    float *TempVektor_2 = NULL;
    float *pVektor = NULL;
    float Operand_1, Operand_2 = 0.f;
    int Temp_V_Len_1 = 0;
    int Temp_V_Len_2 = 0;
    int Type_Res;
    float Skalar_Res = 0;
    int x = 0;
    int Man_Vek = 0;
#ifdef DEBUGMODE
    char buf[400];
    char value[100];
#endif

    Stackinit_Eval();

    while (Done != true)
    {
        switch (pListPostfix[CountPostfix].Priority)
        {
        case 0: // operand: push onto stack
            //scalar
            if (pListPostfix[CountPostfix].Type == SKALAR)
            {
                if (!strcmp(pListPostfix[CountPostfix].Item, SKALAR_1))
                {
                    if (pSkalar_1)
                        Temp = *pSkalar_1;
                    else
                        Temp = 0.f;
                }
                if (!strcmp(pListPostfix[CountPostfix].Item, SKALAR_2))
                {
                    if (pSkalar_2)
                        Temp = *pSkalar_2;
                    else
                        Temp = 0.f;
                }
                if (strcmp(pListPostfix[CountPostfix].Item, SKALAR_1) != 0)
                {
                    if (strcmp(pListPostfix[CountPostfix].Item, SKALAR_2) != 0)
                    {
                        Temp = (float)atof(pListPostfix[CountPostfix].Item);
                    }
                }
                Push(Temp, SKALAR, NULL, 0);
                CountPostfix++;
                break;
            }

            //vector
            if (pListPostfix[CountPostfix].Type == VEKTOR)
            {
                if (!strcmp(pListPostfix[CountPostfix].Item, VEKTOR_1))
                {
                    pVektor = pVektor_1;
                }
                if (!strcmp(pListPostfix[CountPostfix].Item, VEKTOR_2))
                {
                    pVektor = pVektor_2;
                }
                if (strstr(pListPostfix[CountPostfix].Item, MANUAL_V))
                {
                    Man_Vek = atoi((pListPostfix[CountPostfix].Item) + strlen(MANUAL_V));
                    pVektor = &pMan_Vektor[Man_Vek * Vek_Len];
                }
                if (!strcmp(pListPostfix[CountPostfix].Item, EINHEITS_V))
                {
                    pVektor = pEinh_Vektor;
                    for (x = 0; x < Vek_Len; x++)
                        pVektor[x] = 1;
                }
                Temp = 0;
                Push(Temp, VEKTOR, pVektor, Vek_Len);
                CountPostfix++;
                break;
            }

        case 2: // operator
        case 3:
        case 4:
        case 5:
            //no function
            if (pListPostfix[CountPostfix].Priority != 5)
                Operand_2 = Pop(&Op_Type_2, &TempVektor_2, &Temp_V_Len_2);

            else
                Op_Type_2 = false;

            Operand_1 = Pop(&Op_Type_1, &TempVektor_1, &Temp_V_Len_1);
            Legal = CheckOp(Op_Type_1, Op_Type_2,
                            pListPostfix[CountPostfix].Token, &Operation, &Type_Res);
            if (!Legal)
            {
                module->sendError("ERROR: non legal operation in expression");
                return (0);
            }

            if (!PerformOperation(module, Operand_1, Operand_2, TempVektor_1,
                                  TempVektor_2, Vek_Len, Operation, pListPostfix[CountPostfix].Token,
                                  pListPostfix[CountPostfix - 1].Item, &Type_Res, &Skalar_Res, Vektor_Res))
                return (0);

            Push(Skalar_Res, Type_Res, *Vektor_Res, Vek_Len);
            if (Type_Res == VEKTOR)
                (*Vektor_Res) = (*Vektor_Res) + Vek_Len;
            CountPostfix++;
            break;

        case EOL: // end of list: result is on stack
            (*Vektor_Res) = (*Vektor_Res) - Vek_Len;
            *Result_Type = Type_Res;
            *Result_Skalar = Skalar_Res;
            delete (item_eval);
            Done = true;
            break;
        }
    }

#ifdef DEBUGMODE
    if (Array_Len == 1)
    {
        switch (Type_Res)
        {
        case SKALAR:
            strcpy(buf, "Result (scalar): ");
            sprintf(value, "%f", Skalar_Res);
            strcat(buf, value);
            module->sendInfo(buf);
            break;

        case VEKTOR:
            strcpy(buf, "Result (vector): [");
            sprintf(value, "%f,%f,%f]", ((*Vektor_Res)[0]), ((*Vektor_Res)[1]), ((*Vektor_Res)[2]));
            strcat(buf, value);
            module->sendInfo(buf);
            break;
        }
    }
#endif

    Stack_Eval_Free();
    return (1);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// PerformOperation: perform one operation                                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::PerformOperation(Calc *module, float Op_1, float Op_2, float *V_1, float *V_2,
                            int V_Len, int Operation, int Token,
                            char *input, int *Type_Res,
                            float *S_Res, float **V_Res)
{
    int Count = 0;
    int Count_max_min = 0;
    float puffer;
    float length;
#ifdef DEBUGMODE
    char buf[400];
    char value[100];
    char pos[20];
    int position = 0;
#endif

    switch (Operation)
    {
    case VEK_VEK:
        switch (Token)
        {
        case PLUS:
            for (Count = 0; Count < V_Len; Count++)
            {
                (*V_Res)[Count] = V_1[Count] + V_2[Count];
            }
            *Type_Res = VEKTOR;
            break;

        case MINUS:
            for (Count = 0; Count < V_Len; Count++)
            {
                (*V_Res)[Count] = V_1[Count] - V_2[Count];
            }
            *Type_Res = VEKTOR;
            break;

        case MAL:
            for (Count = 0; Count < V_Len; Count++)
            {
                *S_Res = (*S_Res) + (V_1[Count] * V_2[Count]);
            }
            *Type_Res = SKALAR;
            break;

        case VEK_PROD:
            if (V_Len != 3)
            {
                module->sendError("ERROR: `Kreuzprodukt` only in R3");
                return (0);
            }
            (*V_Res)[0] = V_1[1] * V_2[2] - V_1[2] * V_2[1];
            (*V_Res)[1] = V_1[2] * V_2[0] - V_1[0] * V_2[2];
            (*V_Res)[2] = V_1[0] * V_2[1] - V_1[1] * V_2[0];
            *Type_Res = VEKTOR;
            break;
        }
        break;

    case VEK_SKA:
        switch (Token)
        {
        case MAL:
            for (Count = 0; Count < V_Len; Count++)
            {
                (*V_Res)[Count] = V_1[Count] * Op_2;
            }
            *Type_Res = VEKTOR;
            break;

        case GETEILT:
            if (Op_2)
            {
                for (Count = 0; Count < V_Len; Count++)
                {
                    (*V_Res)[Count] = V_1[Count] / Op_2;
                }
            }
            else
            {
                module->sendError("ERROR: division by zero");
                return (0);
            }
            *Type_Res = VEKTOR;
            break;
        }
        break;

    case SKA_VEK:
        switch (Token)
        {
        case MAL:
            for (Count = 0; Count < V_Len; Count++)
            {
                (*V_Res)[Count] = V_2[Count] * Op_1;
            }
            *Type_Res = VEKTOR;
            break;
        }
        break;

    case SKA_SKA:
        switch (Token)
        {
        case PLUS:
            *S_Res = Op_1 + Op_2;
            break;

        case MINUS:
            *S_Res = Op_1 - Op_2;
            break;

        case MAL:
            *S_Res = Op_1 * Op_2;
            break;

        case GETEILT:
            if (Op_2)
                *S_Res = Op_1 / Op_2;
            else
            {
                module->sendWarning("ERROR: division by zero");
                //return(0);
                *S_Res = 0.;
            }
            break;

        case HOCH:
            *S_Res = pow(Op_1, Op_2);
            break;

        case WURZEL:
            *S_Res = pow(Op_1, 1 / Op_2);
            break;
        }
        *Type_Res = SKALAR;
        break;

    case VEK:
        switch (Token)
        {
        case NEG:
            for (Count = 0; Count < V_Len; Count++)
            {
                (*V_Res)[Count] = (-1) * V_1[Count];
            }
            *Type_Res = VEKTOR;
            break;

        case VLEN:
            *S_Res = 0;
            for (Count = 0; Count < V_Len; Count++)
            {
                *S_Res = *S_Res + (V_1[Count] * V_1[Count]);
            }
            *S_Res = sqrt(*S_Res);
            *Type_Res = SKALAR;
            break;

        case COMP_1:
            *S_Res = V_1[0];
            *Type_Res = SKALAR;
            break;

        case COMP_2:
            *S_Res = V_1[1];
            *Type_Res = SKALAR;
            break;

        case COMP_3:
            *S_Res = V_1[2];
            *Type_Res = SKALAR;
            break;

        case MAX:
            if (!strcmp(input, VEKTOR_1))
            {
                puffer = 0;
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    length = sqrt((u1_in[Count_max_min] * u1_in[Count_max_min]) + (v1_in[Count_max_min] * v1_in[Count_max_min]) + (w1_in[Count_max_min] * w1_in[Count_max_min]));

                    if (length > puffer)
                    {
                        puffer = length;
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
//write message with maximum value
#ifdef DEBUGMODE
                if (!minmax)
                {
                    strcpy(buf, "maximum length of v1: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }

            if (!strcmp(input, VEKTOR_2))
            {
                puffer = 0;
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    length = sqrt((u2_in[Count_max_min] * u2_in[Count_max_min]) + (v2_in[Count_max_min] * v2_in[Count_max_min]) + (w2_in[Count_max_min] * w2_in[Count_max_min]));

                    if (length > puffer)
                    {
                        puffer = length;
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
#ifdef DEBUGMODE
                //write message with maximum value
                if (!minmax)
                {
                    strcpy(buf, "maximum length of v2: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }
            module->sendError("ERROR: max() can only be used with s1, s2, v1 or v2");
            return (0);

        case MIN:
            if (!strcmp(input, VEKTOR_1))
            {
                puffer = sqrt((u1_in[0] * u1_in[0]) + (v1_in[0] * v1_in[0]) + (w1_in[0] * w1_in[0]));
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    length = sqrt((u1_in[Count_max_min] * u1_in[Count_max_min]) + (v1_in[Count_max_min] * v1_in[Count_max_min]) + (w1_in[Count_max_min] * w1_in[Count_max_min]));

                    if (length < puffer)
                    {
                        puffer = length;
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
#ifdef DEBUGMODE
                //write message with minimum value
                if (!minmax)
                {
                    strcpy(buf, "minimum length of v1: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }

            if (!strcmp(input, VEKTOR_2))
            {
                puffer = sqrt((u1_in[0] * u1_in[0]) + (v1_in[0] * v1_in[0]) + (w1_in[0] * w1_in[0]));
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    length = sqrt((u2_in[Count_max_min] * u2_in[Count_max_min]) + (v2_in[Count_max_min] * v2_in[Count_max_min]) + (w2_in[Count_max_min] * w2_in[Count_max_min]));

                    if (length < puffer)
                    {
                        puffer = length;
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
#ifdef DEBUGMODE
                //write message with minimum value
                if (!minmax)
                {
                    strcpy(buf, "minimum length of v2: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }
            module->sendError("ERROR: min() can only be used with s1, s2, v1 or v2");
            return (0);
        }
        break;

    case SKA:
        switch (Token)
        {
        case SIN:
            *S_Res = sin(Op_1);
            break;

        case COS:
            *S_Res = cos(Op_1);
            break;

        case TAN:
            *S_Res = tan(Op_1);
            break;

        case ATAN:
            *S_Res = atan(Op_1);
            break;

        case LOG:
            if (Op_1 >= 0)
                *S_Res = log(Op_1);
            else
            {
                module->sendError("ERROR: log on negativ operand");
                return (0);
            }
            break;

        case EXP:
            *S_Res = exp(Op_1);
            break;

        case NEG:
            *S_Res = (-Op_1);
            break;

        case MAX:
            if (!strcmp(input, SKALAR_1))
            {
                puffer = 0;
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    if (s1_in[Count_max_min] > puffer)
                    {
                        puffer = s1_in[Count_max_min];
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
#ifdef DEBUGMODE
                //write message with maximum value
                if (!minmax)
                {
                    strcpy(buf, "maximum value of s1: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }

            if (!strcmp(input, SKALAR_2))
            {
                puffer = 0;
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    if (s2_in[Count_max_min] > puffer)
                    {
                        puffer = s2_in[Count_max_min];
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
#ifdef DEBUGMODE
                //write message with maximum value
                if (!minmax)
                {
                    strcpy(buf, "maximum value of s2: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }
            module->sendError("ERROR: max() can only be used with s1, s2, v1 or v2");
            return (0);

        case MIN:
            if (!strcmp(input, SKALAR_1))
            {
                puffer = s1_in[0];
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    if (s1_in[Count_max_min] < puffer)
                    {
                        puffer = s1_in[Count_max_min];
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
#ifdef DEBUGMODE
                //write message with minimum value
                if (!minmax)
                {
                    strcpy(buf, "minimum value of s1: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }

            if (!strcmp(input, SKALAR_2))
            {
                puffer = s2_in[0];
                for (Count_max_min = 0; Count_max_min < Array_Len; Count_max_min++)
                {
                    if (s2_in[Count_max_min] < puffer)
                    {
                        puffer = s2_in[Count_max_min];
#ifdef DEBUGMODE
                        position = Count_max_min + 1;
#endif
                    }
                }
                *S_Res = puffer;
//write message with minimum value
#ifdef DEBUGMODE
                if (!minmax)
                {
                    strcpy(buf, "minimum value of s2: ");
                    sprintf(value, "%f", puffer);
                    strcat(buf, value);
                    strcat(buf, " (Pos.: ");
                    sprintf(pos, "%d", position);
                    strcat(buf, pos);
                    strcat(buf, ")");
                    module->sendInfo(buf);
                }
#endif
                break;
            }
            module->sendError("ERROR: min() can only be used with s1, s2, v1 or v2");
            return (0);
        }
        *Type_Res = SKALAR;
        break;
    }

    return (1);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// CheckOp: check if operation is legal                                     //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::CheckOp(int Type_1, int Type_2, int Operator, int *Operation,
                   int *Ergebnis)
{

    switch (Type_1)
    {
    case VEKTOR:
        switch (Type_2)
        {
        case VEKTOR:
            *Operation = VEK_VEK;
            break;
        case SKALAR:
            *Operation = VEK_SKA;
            break;
        default:
            *Operation = VEK;
            break;
        }
        break;

    case SKALAR:
        switch (Type_2)
        {
        case VEKTOR:
            *Operation = SKA_VEK;
            break;
        case SKALAR:
            *Operation = SKA_SKA;
            break;
        default:
            *Operation = SKA;
            break;
        }
        break;
    }

    *Ergebnis = pLegal[(2 * (*Operation) * (MAX_OPERATIONS + MAX_FUNKT)) + Operator + (MAX_OPERATIONS + MAX_FUNKT)];
    return (pLegal[(2 * (*Operation) * (MAX_OPERATIONS + MAX_FUNKT)) + Operator]);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// CheckInputs: check inputs (what a surprise)                              //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

int CCalc::CheckInputs(Calc *module, int Code_s1, int Code_s2, int Code_v1, int Code_v2,
                       int i_s1, int j_s1, int k_s1,
                       int i_s2, int j_s2, int k_s2,
                       int i_v1, int j_v1, int k_v1,
                       int i_v2, int j_v2, int k_v2,
                       int d_s1, int d_s2, int d_v1, int d_v2,
                       int *A_Length, int *s_u,
                       int *i_dimen, int *j_dimen, int *k_dimen, int *dimen)
{
    int CheckCode;
    float Checksum;

    CheckCode = Code_s1 + Code_s2 + Code_v1 + Code_v2;

    switch (CheckCode)
    {
    case 0: //no input attached
        module->sendError("WARNING: No input attached");
        *A_Length = 1;
        *s_u = UNSTR;
        *dimen = *A_Length;
        break;

    case 1: // OK (STR)
    case 2:
    case 3:
    case 4: // check, if dimensions of i-compound are equal
        Checksum = (float)((i_s1 + i_s2 + i_v1 + i_v2) / ((Code_s1 + Code_s2 + Code_v1 + Code_v2) / STR));
        if (fabs(Checksum - Maximum(i_s1, i_s2, i_v1, i_v2)) >= 0.25)
        {
            module->sendError("ERROR: Different input dimensions");
            return (0);
        }
        // check, if dimensions of j-compound are equal
        Checksum = (float)((j_s1 + j_s2 + j_v1 + j_v2) / ((Code_s1 + Code_s2 + Code_v1 + Code_v2) / STR));
        if (fabs(Checksum - Maximum(j_s1, j_s2, j_v1, j_v2)) >= 0.25)
        {
            module->sendError("ERROR: Different input dimensions");
            return (0);
        }
        // check, if dimensions of k-compound are equal
        Checksum = (float)((k_s1 + k_s2 + k_v1 + k_v2) / ((Code_s1 + Code_s2 + Code_v1 + Code_v2) / STR));
        if (fabs(Checksum - Maximum(k_s1, k_s2, k_v1, k_v2)) >= 0.25)
        {
            module->sendError("ERROR: Different input dimensions");
            return (0);
        }
        *A_Length = Maximum(i_s1, i_s2, i_v1, i_v2) * Maximum(j_s1, j_s2, j_v1, j_v2) * Maximum(k_s1, k_s2, k_v1, k_v2);
        *s_u = STR;
        *i_dimen = Maximum(i_s1, i_s2, i_v1, i_v2);
        *j_dimen = Maximum(j_s1, j_s2, j_v1, j_v2);
        *k_dimen = Maximum(k_s1, k_s2, k_v1, k_v2);
        break;

    case 5: // OK (UNSTR)
    case 10:
    case 15:
    case 20: // check, if dimensions of input are equal
        Checksum = (float)((d_s1 + d_s2 + d_v1 + d_v2) / ((Code_s1 + Code_s2 + Code_v1 + Code_v2) / UNSTR));
        if ((fabs(Checksum - Maximum(d_s1, d_s2, d_v1, d_v2))) >= 0.25)
        {
            //module->sendError("ERROR: Different input dimensions, cutting to shortest dimension");
            *A_Length = Minimum(d_s1, d_s2, d_v1, d_v2);
            *s_u = UNSTR;
            *dimen = *A_Length;
            module->sendError("WARNING: Different input dimensions (s1:%d, s2:%d, v1:%d, v2:%d), cutting to shortest dimension (%d)", d_s1, d_s2, d_v1, d_v2, (*A_Length));
            return (1);
        }
        *A_Length = Maximum(d_s1, d_s2, d_v1, d_v2);
        *s_u = UNSTR;
        *dimen = *A_Length;
        break;

    case 6: // not OK
    case 7:
    case 8:
    case 11:
    case 12:
    case 16:
        module->sendError("ERROR: Structured / unstructured mismatch");
        return (0);
    }

    return (1);
}

MODULE_MAIN(Tools, Calc)
