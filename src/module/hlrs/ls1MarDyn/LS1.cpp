/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LS1.h"
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <api/coModule.h>
#include <limits.h>
#include <api/coFeedback.h>
#include <iostream>

using namespace std;

#include <steereo/steereoClientSteering.h>
#include <steereo/steereoSocketCommunicator.h>
#include <steereo/steerParameterClient.h>

MoleculeRenderer::MoleculeRenderer(int argc, char *argv[])
    : coModule(argc, argv, "MarDyn LS1 - MoleculeRenderer")
{
    //Port fuer die Koordinaten der Molekuele
    renderedmolecules = addOutputPort("Molekuele", "Points", "points");
    //Molekuelgeschwindigkeiten
    molecule_velocity = addOutputPort("Geschwindigkeit", "Vec3", "velocity");
    //kinetische Energie jedes Molekuels
    molecule_kinetic_energy = addOutputPort("KinEnergie", "Float", "E_kin");
    //Druck des Molekuelcontainers
    domain_pressure = addOutputPort("Druck", "Float", "pressure");
    //Temperatur des Molekuelcontainers
    domain_temp = addOutputPort("Temperatur", "Float", "temperature");
    //begrenzende Box
    boundingbox = addOutputPort("Flaeche", "UnstructuredGrid", "mesh");
    //Port fuer das Renderer-Plugin
    Tempport = addOutputPort("AdjustTemp", "Float", "temp");

    //Parameter fuer die Zeitschritte die dargestellt werden sollen
    _p_NumberOfTimeSteps = addInt32Param("numSteps", "Number of steps to read");
    _p_NumberOfTimeSteps->setValue(1);

    //Parameter fuer die zu rendernden Zeitschritte
    _p_renderedTimeSteps = addInt32Param("rendSteps", "Intervall of timesteps to render (1 means every timestep");
    _p_renderedTimeSteps->setValue(1);

    //Parameter zum aendern der Temperatur
    _p_chTemp = addFloatSliderParam("Temp", "aenderbare Temperatur");
    MinTemp = 0.1;
    MaxTemp = 10.0;
    ValTemp = 0.7;
    _p_chTemp->setValue(MinTemp, MaxTemp, ValTemp);

    _p_ipaddress = addStringParam("IPAddress", "IPAddress to connect with Simulation");
    _p_ipaddress->setValue("127.0.0.1");
    _p_tcpport = addStringParam("TCPPort", "Port to connect with Simulation");
    _p_tcpport->setValue("44445");
}

int MoleculeRenderer::compute(const char *)
{
    coDoFloat *changeTemp;
    float *x, *y, *z, *vecx, *vecy, *vecz, *p, *T, *ekin, *f1, *f2, *f3, *TempValue;
    unsigned long timestep = 0, laeufer = 0;
    unsigned long DATASIZE, Timesteps, Counter, aTS, zielwert;
    int *el, *cl, Simschritte, Zeitschritte;
    float XBox, YBox, ZBox, Pressure, TV;

    //Steereo-Funktionalitaet
    SteereoClientSteering *clientSteer;
    SteerParameterClient *paramClient = new SteerParameterClient();
    clientSteer = new SteereoClientSteering;
    SteereoLogger::setOutputLevel(2);
    SteereoSocketCommunicator *clientComm = new SteereoSocketCommunicator;
    clientSteer->startConnection(clientComm, _p_ipaddress->getValue(), _p_tcpport->getValue());
    paramClient->setClient(clientSteer);
    paramClient->addAction();

    //aktueller Zeitschritt in der Simulation
    paramClient->registerScalarParameter("aktuellerTS", &aTS);
    //Menge der Molekuele in der Simulation registrieren
    paramClient->registerScalarParameter("datasize", &DATASIZE);
    //Menge der Zeitschritte in der Simulation registrieren
    paramClient->registerScalarParameter("zeitschritte", &Timesteps);
    //Laenge der BoundingBox in x-Richtung registrieren
    paramClient->registerScalarParameter("Scalar_x", &XBox);
    //Laenge der BoundingBox in y-Richtung registrieren
    paramClient->registerScalarParameter("Scalar_y", &YBox);
    //Laenge der BoundingBox in z-Richtung registrieren
    paramClient->registerScalarParameter("Scalar_z", &ZBox);
    clientSteer->startAccepting();

    //Abholen der Parameter
    paramClient->requestGetParameter("zeitschritte");
    paramClient->requestGetParameter("datasize");
    paramClient->requestGetParameter("Scalar_x");
    paramClient->requestGetParameter("Scalar_y");
    paramClient->requestGetParameter("Scalar_z");
    while (!paramClient->areAllParametersUpdated())
    {
        sleep(1);
    }

    //Deklaration der DATASIZE-abhaengigen Parameter
    float *myDataArray_id = new float[DATASIZE];
    float *myDataArrayx = new float[DATASIZE];
    float *myDataArrayy = new float[DATASIZE];
    float *myDataArrayz = new float[DATASIZE];
    float *myDataArrayvx = new float[DATASIZE];
    float *myDataArrayvy = new float[DATASIZE];
    float *myDataArrayvz = new float[DATASIZE];
    float *myDataArrayEkin = new float[DATASIZE];

    //Molekueltyp registrieren
    paramClient->registerArrayParameter("molecule", myDataArray_id, DATASIZE);
    //x-Werte registrieren
    paramClient->registerArrayParameter("Arrayx", myDataArrayx, DATASIZE);
    //y-Werte registrieren
    paramClient->registerArrayParameter("Arrayy", myDataArrayy, DATASIZE);
    //z-Werte registrieren
    paramClient->registerArrayParameter("Arrayz", myDataArrayz, DATASIZE);
    //Geschwindigkeit in x-Richtung registrieren
    paramClient->registerArrayParameter("Arrayvx", myDataArrayvx, DATASIZE);
    //Geschwindigkeit in y-Richtung registrieren
    paramClient->registerArrayParameter("Arrayvy", myDataArrayvy, DATASIZE);
    //Geschwindigkeit in z-Richtung registrieren
    paramClient->registerArrayParameter("Arrayvz", myDataArrayvz, DATASIZE);
    //Registrieren der kinetischen Energie
    paramClient->registerArrayParameter("E_kin", myDataArrayEkin, DATASIZE);
    //Registrieren des Drucks
    paramClient->registerScalarParameter("Scalar_p", &Pressure);
    //Registrieren der Temperatur
    paramClient->registerScalarParameter("Scalar_T", &Temperature);

    Temperature = ValTemp;
    paramClient->requestSetParameter("Scalar_T");
    while (!paramClient->isParameterUpdated("Scalar_T"))
    {
        sleep(1);
    }

    //Anlegen der Objekte...ein Element groeßer als Zeitschritte in der Simulation
    coDistributedObject **object_1 = new coDistributedObject *[Timesteps + 1];
    coDistributedObject **object_2 = new coDistributedObject *[Timesteps + 1];
    coDistributedObject **object_3 = new coDistributedObject *[Timesteps + 1];
    coDistributedObject **object_4 = new coDistributedObject *[Timesteps + 1];
    coDistributedObject **object_5 = new coDistributedObject *[Timesteps + 1];

    //Objekt fuer die bounding box
    coDoUnstructuredGrid *MDLS1_Flaeche = new coDoUnstructuredGrid(boundingbox->getObjName(), 6, 24, 8, 1);
    MDLS1_Flaeche->getAddresses(&el, &cl, &f1, &f2, &f3);

    el[0] = 0;
    el[1] = 4;
    el[2] = 8;
    el[3] = 12;
    el[4] = 16;
    el[5] = 20;

    cl[0] = 0;
    cl[1] = 1;
    cl[2] = 2;
    cl[3] = 3;
    cl[4] = 4;
    cl[5] = 5;
    cl[6] = 6;
    cl[7] = 7;
    cl[8] = 1;
    cl[9] = 2;
    cl[10] = 5;
    cl[11] = 6;
    cl[12] = 2;
    cl[13] = 3;
    cl[14] = 6;
    cl[15] = 7;
    cl[16] = 0;
    cl[17] = 3;
    cl[18] = 4;
    cl[19] = 7;
    cl[20] = 0;
    cl[21] = 1;
    cl[22] = 4;
    cl[23] = 5;

    f1[0] = 0;
    f2[0] = 0;
    f3[0] = ZBox;
    f1[1] = 0;
    f2[1] = 0;
    f3[1] = 0;
    f1[2] = XBox;
    f2[2] = 0;
    f3[2] = 0;
    f1[3] = XBox;
    f2[3] = 0;
    f3[3] = ZBox;
    f1[4] = 0;
    f2[4] = YBox;
    f3[4] = ZBox;
    f1[5] = 0;
    f2[5] = YBox;
    f3[5] = 0;
    f1[6] = XBox;
    f2[6] = YBox;
    f3[6] = 0;
    f1[7] = XBox;
    f2[7] = YBox;
    f3[7] = ZBox;

    /* ------------------ */
    /* BEGIN MAIN ROUTINE */
    /* ------------------ */

    Zeitschritte = _p_NumberOfTimeSteps->getValue();
    Simschritte = _p_renderedTimeSteps->getValue();
    Counter = Zeitschritte * Simschritte;

    while (timestep < Zeitschritte)
    {
        zielwert = 0;
        if (Simschritte == 1)
        {
            for (int j = 0; j < Simschritte; j++)
            {
                char name[2048];
                snprintf(name, sizeof(name), "%s_%lu", renderedmolecules->getObjName(), laeufer);
                coDoPoints *MDLS1 = new coDoPoints(name, DATASIZE);
                snprintf(name, sizeof(name), "%s_%lu", molecule_velocity->getObjName(), laeufer);
                coDoVec3 *MDLS1_vec = new coDoVec3(name, DATASIZE);
                snprintf(name, sizeof(name), "%s_%lu", molecule_kinetic_energy->getObjName(), laeufer);
                coDoFloat *MDLS1_kin = new coDoFloat(name, DATASIZE);
                snprintf(name, sizeof(name), "%s_%lu", domain_pressure->getObjName(), laeufer);
                coDoFloat *MDLS1_press = new coDoFloat(name, DATASIZE);
                snprintf(name, sizeof(name), "%s_%lu", domain_temp->getObjName(), laeufer);
                coDoFloat *MDLS1_temp = new coDoFloat(name, DATASIZE);

                MDLS1->getAddresses(&x, &y, &z);
                MDLS1_vec->getAddresses(&vecx, &vecy, &vecz);
                MDLS1_kin->getAddress(&ekin);
                MDLS1_press->getAddress(&p);
                MDLS1_temp->getAddress(&T);

                paramClient->requestGetParameter("Arrayx");
                paramClient->requestGetParameter("Arrayy");
                paramClient->requestGetParameter("Arrayz");
                paramClient->requestGetParameter("Arrayvx");
                paramClient->requestGetParameter("Arrayvy");
                paramClient->requestGetParameter("Arrayvz");
                paramClient->requestGetParameter("E_kin");
                paramClient->requestGetParameter("Scalar_p");
                paramClient->requestGetParameter("Scalar_T");
                while (!paramClient->areAllParametersUpdated())
                {
                    sleep(1);
                }

                for (int i = 0; i < DATASIZE; i++)
                {
                    x[i] = myDataArrayx[i];
                    y[i] = myDataArrayy[i];
                    z[i] = myDataArrayz[i];
                    vecx[i] = myDataArrayvx[i];
                    vecy[i] = myDataArrayvy[i];
                    vecz[i] = myDataArrayvz[i];
                    ekin[i] = myDataArrayEkin[i];
                    p[i] = Pressure;
                    T[i] = (float)Temperature;
                }

                object_1[laeufer] = MDLS1;
                object_2[laeufer] = MDLS1_vec;
                object_3[laeufer] = MDLS1_press;
                object_4[laeufer] = MDLS1_temp;
                object_5[laeufer] = MDLS1_kin;
            }
        }
        else
        {
            char name[2048];
            snprintf(name, sizeof(name), "%s_%lu", renderedmolecules->getObjName(), laeufer);
            coDoPoints *MDLS1 = new coDoPoints(name, DATASIZE);
            snprintf(name, sizeof(name), "%s_%lu", molecule_velocity->getObjName(), laeufer);
            coDoVec3 *MDLS1_vec = new coDoVec3(name, DATASIZE);
            snprintf(name, sizeof(name), "%s_%lu", molecule_kinetic_energy->getObjName(), laeufer);
            coDoFloat *MDLS1_kin = new coDoFloat(name, DATASIZE);
            snprintf(name, sizeof(name), "%s_%lu", domain_pressure->getObjName(), laeufer);
            coDoFloat *MDLS1_press = new coDoFloat(name, DATASIZE);
            snprintf(name, sizeof(name), "%s_%lu", domain_temp->getObjName(), laeufer);
            coDoFloat *MDLS1_temp = new coDoFloat(name, DATASIZE);

            MDLS1->getAddresses(&x, &y, &z);
            MDLS1_vec->getAddresses(&vecx, &vecy, &vecz);
            MDLS1_kin->getAddress(&ekin);
            MDLS1_press->getAddress(&p);
            MDLS1_temp->getAddress(&T);

            paramClient->requestGetParameter("Arrayx");
            paramClient->requestGetParameter("Arrayy");
            paramClient->requestGetParameter("Arrayz");
            paramClient->requestGetParameter("Arrayvx");
            paramClient->requestGetParameter("Arrayvy");
            paramClient->requestGetParameter("Arrayvz");
            paramClient->requestGetParameter("E_kin");
            paramClient->requestGetParameter("Scalar_p");
            paramClient->requestGetParameter("Scalar_T");
            while (!paramClient->areAllParametersUpdated())
            {
                sleep(1);
            }

            for (int i = 0; i < DATASIZE; i++)
            {
                x[i] = myDataArrayx[i];
                y[i] = myDataArrayy[i];
                z[i] = myDataArrayz[i];
                vecx[i] = myDataArrayvx[i];
                vecy[i] = myDataArrayvy[i];
                vecz[i] = myDataArrayvz[i];
                ekin[i] = myDataArrayEkin[i];
                p[i] = Pressure;
                T[i] = (float)Temperature;
            }

            object_1[laeufer] = MDLS1;
            object_2[laeufer] = MDLS1_vec;
            object_3[laeufer] = MDLS1_press;
            object_4[laeufer] = MDLS1_temp;
            object_5[laeufer] = MDLS1_kin;

            paramClient->requestGetParameter("aktuellerTS");
            while (!paramClient->isParameterUpdated("aktuellerTS"))
            {
                sleep(1);
            }
            zielwert = aTS + Simschritte;
            while (aTS < zielwert)
            {
                paramClient->requestGetParameter("aktuellerTS");
                while (!paramClient->isParameterUpdated("aktuellerTS"))
                {
                    sleep(1);
                }
            }
        }
        laeufer++;
        timestep++;
    }

    object_1[laeufer] = NULL;
    object_2[laeufer] = NULL;
    object_3[laeufer] = NULL;
    object_4[laeufer] = NULL;
    object_5[laeufer] = NULL;

    if (Tempport->getObjName())
    {
        changeTemp = new coDoFloat(Tempport->getObjName(), 1);
        changeTemp->getAddress(&TempValue);
        TV = _p_chTemp->getValue();
        TempValue = &TV;

        coFeedback feedback("LS1");
        feedback.addPara(_p_chTemp);
        feedback.apply(changeTemp);
        Tempport->setCurrentObject(changeTemp);
    }

    coDoSet *Punkte = new coDoSet(renderedmolecules->getObjName(), object_1);
    coDoSet *Velo = new coDoSet(molecule_velocity->getObjName(), object_2);
    coDoSet *Druck = new coDoSet(domain_pressure->getObjName(), object_3);
    coDoSet *Temperatur = new coDoSet(domain_temp->getObjName(), object_4);
    coDoSet *KinEnergie = new coDoSet(molecule_kinetic_energy->getObjName(), object_5);

    char buf[16];
    snprintf(buf, 16, "%d", Simschritte);
    Punkte->addAttribute("TIMESTEP", buf);
    Velo->addAttribute("TIMESTEP", buf);
    Druck->addAttribute("TIMESTEP", buf);
    Temperatur->addAttribute("TIMESTEP", buf);
    KinEnergie->addAttribute("TIMESTEP", buf);

    renderedmolecules->setCurrentObject(Punkte);
    molecule_velocity->setCurrentObject(Velo);
    molecule_kinetic_energy->setCurrentObject(KinEnergie);
    domain_pressure->setCurrentObject(Druck);
    domain_temp->setCurrentObject(Temperatur);
    boundingbox->setCurrentObject(MDLS1_Flaeche);

    return CONTINUE_PIPELINE;

    clientSteer->closeConnections();
}

MoleculeRenderer::~MoleculeRenderer()
{
}

void MoleculeRenderer::param(const char *name, bool inMapLoading)
{
    if (strcmp(name, _p_chTemp->getName()) == 0)
    {
        ValTemp = _p_chTemp->getValue();
    }
}

MODULE_MAIN(HLRS, MoleculeRenderer)
