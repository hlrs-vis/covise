/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:  COVISE Calc application module                           **
 **                                                                        **
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
 **        8.12.2000                                                       **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

const unsigned MAXLEN = 80; // Max. Länge eines Ausdrucks
const unsigned MAXITEM = 20; // Max. Länge eines Ausdrucks
const unsigned EOL = 255; // Ende der Liste
const unsigned STR = 1; // structured
const unsigned UNSTR = 5; // unstructured

const unsigned NUMSCALAR = 2;
const unsigned NUMVECTOR = 2;

//Zulässige Funktionen
const char *SINUS = "sin";
const char *COSINUS = "cos";
const char *TANGENS = "tan";
const char *ARCTAN = "atan";
const char *LOGARITH = "log";
const char *EXPON = "exp";
const char *NEGATIV = "neg";
const char *VEKTOR_LEN = "vlen";
const char *COMPOUND_1 = "comp1";
const char *COMPOUND_2 = "comp2";
const char *COMPOUND_3 = "comp3";
const char *MAXIMUM = "max";
const char *MINIMUM = "min";
//bei neuen Funktionen MAX_FUNKT entsprechend erhoehen

const unsigned MAX_FUNKT = 13; //Anzahl der erlaubten Funktionen
const unsigned MAX_OPERATIONS = 7; //Anzahl der erlaubten Operationen

//Zulässige Bezeichnungen im Ausdruck
const char *EINHEITS_V = "e";
const char *VEKTOR_1 = "v1";
const char *VEKTOR_2 = "v2";
const char *SKALAR_1 = "s1";
const char *SKALAR_2 = "s2";
const char *MANUAL_V = "man"; //only for internal use

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Typedefs                                                                 //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

typedef struct LIST //Item-Liste
{
    char *Item; // Einzelner Ausdruck
    int Priority; // Priorität
    int Type; // VEKTOR oder SKALAR
    int Token; // PLUS, MINUS, MAL, ...
} LIST;

typedef struct NODE //Stack für InfixToPostfix
{
    char Item[MAXITEM]; // Einzelner Ausdruck
    int Operator; // PLUS, MINUS, MAL, ...
    int Priority; // Priorität
    struct NODE *next; // Nächster Eintrag im Stack
} NODE;

typedef struct NODE_EVAL //Stack für Evaluate
{
    float Skalar; // Skalarwert
    float *pVektor; // Pointer auf den Vektor
    int Type; // VEKTOR oder SKALAR
    int V_Len; // Vektorlänge
    struct NODE_EVAL *next; // Nächster Eintrag im Stack
} NODE_EVAL;

class Calc : public coSimpleModule
{
    //neu
public:
    coInputPort *p_sInData1;
    coInputPort *p_sInData2;
    coInputPort *p_vInData1;
    coInputPort *p_vInData2;
    coOutputPort *p_sOutData1;
    coOutputPort *p_vOutData2;
    coStringParam *p_expression;
    virtual int compute(const char *port);

    Calc(int argc, char *argv[]);
    //    void run() { Covise::main_loop(); }

    virtual ~Calc()
    {
    }

private:
    virtual void preHandleObjects(coInputPort **in_ports);
    virtual void copyAttributesToOutObj(coInputPort **, coOutputPort **, int);
};

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Definition of Class CCalc                                                //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

class CCalc
{
protected:
private:
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    //  Functions                                                               //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////

    void BuildItemList(char *StringExpr);
    void ReadItem(char *String, int *Priority, int *Type, int *Op,
                  int *Space);
    char *LowerCase(char *string);
    int Maximum(int Zahl1, int Zahl2, int Zahl3, int Zahl4);
    int Minimum(int Zahl1, int Zahl2, int Zahl3, int Zahl4);
    void FreeMemory();
    void DeleteItemList();
    void DeletePostfixList();
    int CheckSyntax(Calc *module, int *AnzManVek);
    int GetType(char *Item);

    void Stackinit();
    void Push(char *Item, int Operator, int Priority);
    int Pop(char *PopItem, int *Operator);
    void Stack_Free();

    void Stackinit_Eval();
    void Push(float Item, int Type, float *pVektor, int V_Len);
    float Pop(int *Type, float **pVektor, int *V_Len);
    void Stack_Eval_Free();

    int InfixToPostfix(Calc *module);

    int GetResultType(Calc *module, int *Result_Type, int *TempVek);
    int Evaluate(Calc *module, int *Result_Type, float **Result_Vektor,
                 float *Result_Skalar);
    int PerformOperation(Calc *module, float Op_1, float Op_2, float *V_1,
                         float *V_2, int Vek_Len, int Operation, int Token,
                         char *input, int *Type_Res, float *S_Res, float **V_Res);
    int CheckOp(int Type_1, int Type_2, int Operator, int *Operatio,
                int *Ergebnis);

    int CheckInputs(Calc *module, int s1, int s2, int v1, int v2,
                    int i_s1, int j_s1, int k_s1,
                    int i_s2, int j_s2, int k_s2,
                    int i_v1, int j_v1, int k_v1,
                    int i_v2, int j_v2, int k_v2,
                    int dim_s1, int dim_s2, int dim_v1, int dim_v2,
                    int *A_Length, int *s_u,
                    int *i_dim, int *j_dim, int *k_dim, int *dim);

    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // Global Variables                                                         //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////

    //
    LIST *pList; // Pointer auf Item-Liste
    LIST *pListPostfix; // Pointer auf Item-Liste in Postfixform

    NODE *head; // Pointer auf Kopf des Stapels     (InfixToPostfix)
    NODE *end; // Pointer auf Ende des Stapels     (InfixToPostfix)
    NODE *stack_item; // Pointer auf Element des Stapels  (InfixToPostfix)

    NODE_EVAL *head_eval; // Pointer auf Kopf des Stapels     (Evaluate)
    NODE_EVAL *end_eval; // Pointer auf Ende des Stapels     (Evaluate)
    NODE_EVAL *item_eval; // Pointer auf Element des Stapels  (Evaluate)

    float *pVektor_1; //Pointer auf Vektoreingang 1
    float *pVektor_2; //Pointer auf Vektoreingang 2
    float *pEinh_Vektor; //Pointer auf Einheitsvektor
    float *pMan_Vektor; //Pointer auf manuell eingegebene Vektoren
    int Count_Man_Vektors; //Zähler für manuell eingegebene Vektoren

    float *pSkalar_1; //Pointer auf Skalareingang 1
    float *pSkalar_2; //Pointer auf Skalareingang 2
    int Array_Len; //Anz. der Vektoren /Skalare
    int Vek_Len; //Länge des Vektors

    int Result_Type; //Typ des Ergebnisses
    float *Result_Vektor; //Pointer auf Vektorergebnis
    float Result_Skalar; //Skalarergebnis

    int minmax; //Meldung fuer max/min ausgeben (ja/nein)

    enum Operations
    {
        VEK_VEK,
        VEK_SKA,
        SKA_VEK,
        SKA_SKA,
        VEK,
        SKA
    };

    enum Operators
    {
        PLUS,
        MINUS,
        MAL,
        GETEILT,
        HOCH,
        WURZEL,
        VEK_PROD,
        SIN,
        COS,
        TAN,
        LOG,
        EXP,
        NEG,
        VLEN,
        COMP_1,
        COMP_2,
        COMP_3,
        MAX,
        MIN,
        ATAN,
        //neue Funktionen hier einfuegen
        //.....

        //
        EINH = 251,
        VEK_START = 252,
        VEK_END = 253,
        OPERAND = 254,
        NO_OP = 255
    };

public:
    CCalc(); //constructor
    virtual ~CCalc(); //destructor

    int Compute(Calc *module, char *Ausdruck);

    char Ausdruck[MAXLEN]; //Auszuwertender Ausdruck
};
#endif // _APPLICATION_H
