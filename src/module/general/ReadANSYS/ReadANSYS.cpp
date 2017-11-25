/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coviseCompat.h>
#include <do/coDoText.h>
#include <do/coDoSet.h>
#include "ReadANSYS.h"
#include "ANSYS.h"

const char *NodeChoices[] = { "none" };
const char *ElementChoices[] = {
    "none", "Stresses", "Elastic strains",
    "Plastic strains", "Creep strains",
    "Thermal strains", "Field fluxes",
    "Volume and energies", "Magnetic flux density"
};

const char *SolidComponents[] = {
    "none", "XX", "YY", "ZZ", "XY", "YZ", "ZX",
    "T1", "T2", "T3", "TI", "TIGE"
};
const char *BeamComponents[] = { "none", "Axial", "Yp", "Ym", "Zp", "Zm" };
const char *AxiShellComponents[] = { "none", "Meridional", "ThroughThickness", "Hoop", "Meridional-hoop" };
const char *TopBottomOpts[] = { "Top", "Bottom", "Average" };
const char *ThermalFluxOpts[] = { "none", "QX", "QY", "QZ", "Q" };
const char *VolEnergyOpts[] = { "Volume", "SENE", "KENE" };
const char *MagFluxDensOpts[] = { "B", "BX", "BY", "BZ", "BSUM" };

void
ReadANSYS::postInst()
{
    p_scale_->show();
    coUifPara *feedParams[] = {
        p_rst_,
        p_sol_,
        p_nsol_,
        p_esol_,
        p_stress_,
        p_top_bottom_,
        NULL
    };
    int param;
    for (param = 0; feedParams[param]; ++param)
    {
        allFeedbackParams_.push_back(feedParams[param]);
    }

    // hide params
    hparams_.push_back(h_times_ = new coHideParam(p_times_));
    hparams_.push_back(h_scale_ = new coHideParam(p_scale_));
    hparams_.push_back(h_sol_ = new coHideParam(p_sol_));
    hparams_.push_back(h_nsol_ = new coHideParam(p_nsol_));
    hparams_.push_back(h_esol_ = new coHideParam(p_esol_));
    hparams_.push_back(h_stress_ = new coHideParam(p_stress_));
    hparams_.push_back(h_beam_stress_ = new coHideParam(p_beam_stress_));
    hparams_.push_back(h_axi_shell_stress_ = new coHideParam(p_axi_shell_stress_));
    hparams_.push_back(h_top_bottom_ = new coHideParam(p_top_bottom_));
    hparams_.push_back(h_thermalFlux_ = new coHideParam(p_thermalFlux_));
    hparams_.push_back(h_vol_energy_ = new coHideParam(p_vol_energy_));
    hparams_.push_back(h_mag_flux_dens_ = new coHideParam(p_mag_flux_dens_));
    hparams_.push_back(h_output_node_decode_ = new coHideParam(p_output_node_decode_));
    hparams_.push_back(h_vertex_based_ = new coHideParam(p_vertex_based_));
}

ReadANSYS::ReadANSYS(int argc, char *argv[])
    : coModule(argc, argv, "Read ANSYS data")
    , inMapLoading(false)
{
    oldFileName = NULL;
    open_err_ = 1;
    p_rst_ = addFileBrowserParam("RST_file", "RST file");
    p_rst_->setValue("/var/tmp/rst.rst", "*.rst;*rfl;*rth;*rmg");

    p_times_ = addInt32VectorParam("timeSteps", "select time steps");
    p_times_->setValue(1, 1, 1); // minimum, maximum, jump

    p_scale_ = addFloatParam("ScaleGridDisplacement", "scale grid displacement");
    p_scale_->setValue(1.0);

    p_sol_ = paraSwitch("Solution", "Please enter your choice");
    paraCase("OnlyGeometry");
    paraEndCase();
    paraCase("NodeData");
    p_nsol_ = addChoiceParam("DOF_Solution", "Degrees of freedom");
    p_nsol_->setValue(1, NodeChoices, 0);
    paraEndCase();
    paraCase("ElementData");
    p_esol_ = addChoiceParam("Derived_Solution", "Derived variables");
    p_esol_->setValue(9, ElementChoices, 0);
    p_stress_ = addChoiceParam("SolidShellComponents", "Stress components");
    p_stress_->setValue(12, SolidComponents, 0);
    p_beam_stress_ = addChoiceParam("BeamComponents", "Beam stress components");
    p_beam_stress_->setValue(6, BeamComponents, 0);
    p_axi_shell_stress_ = addChoiceParam("AxiShellComponents", "Axisymmetric-shell stress components");
    p_axi_shell_stress_->setValue(5, AxiShellComponents, 0);
    p_top_bottom_ = addChoiceParam("TopBottom", "Top, bottom, average");
    p_top_bottom_->setValue(3, TopBottomOpts, 0);
    p_thermalFlux_ = addChoiceParam("ThermalFlux", "Thermal flux");
    p_thermalFlux_->setValue(5, ThermalFluxOpts, 0);
    p_vol_energy_ = addChoiceParam("VolEnergy", "Volume and energy");
    p_vol_energy_->setValue(3, VolEnergyOpts, 0);
    p_mag_flux_dens_ = addChoiceParam("MagFluxDens", "Magnetic Flux Density");
    p_mag_flux_dens_->setValue(5, MagFluxDensOpts, 0);
    p_output_node_decode_ = addBooleanParam("OutputNodeDecode",
                                            "Output Node Decode");
    p_output_node_decode_->setValue(0);
    p_vertex_based_ = addBooleanParam("AlwaysVertexBased",
                                      "AlwaysVertexBased");
    p_vertex_based_->setValue(1);
    paraEndCase();
    paraEndSwitch();

    p_file_name_ = addInputPort("FileName", "Text", "File name with extension");
    p_file_name_->setRequired(0);
    p_grid_ = addOutputPort("Grid", "UnstructuredGrid", "output grid");
    p_field_ = addOutputPort("Data", "Float|Vec3|IntArr", "output data");
    p_materials_ = addOutputPort("Materials", "IntArr", "output material labels");
//   p_node_decode_ = addOutputPort("Node_decode_indices", "coDoIntArr",
//                               "Node decode indices");
#ifdef _TEST_RST_FILE_NAME_
    p_outname_ = addOutputPort("TestName", "Text", "File name");
#endif
    new ANSYS();
}

int
ReadANSYS::extractName(std::string &newFileName)
{
    const coDistributedObject *inObj = p_file_name_->getCurrentObject();
    if (inObj == NULL || !inObj->objectOk())
    {
        sendWarning("extractFileName: Got NULL pointer or object is not OK");
        return -1;
    }
    if (!inObj->isType("DOTEXT"))
    {
        sendWarning("extractFileName: Only coDoText is acceptable for input");
        return -1;
    }

    char *text;
    const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
    theText->getAddress(&text);
    istringstream strText(text);
    size_t maxLen = strlen(text) + 1;
    std::vector<char> name(maxLen);
    strText >> &name[0];
    if (maxLen > 1)
    {
        newFileName = &name[0];
    }
    return 0;
}

#include <api/coFeedback.h>

void
ReadANSYS::outputDummies()
{
    coDistributedObject *setList[1] = { NULL };
    coDoSet *dummyGrid = new coDoSet(p_grid_->getObjName(), setList);
    coDoSet *dummyData = new coDoSet(p_field_->getObjName(), setList);
    coDoSet *dummyMat = new coDoSet(p_materials_->getObjName(), setList);
    /*
      if(p_file_name_->isConnected()){
         coFeedback feedback("SCA");
         addFeedbackParams(feedback,allFeedbackParams_);
         feedback.apply(dummyGrid);
      }
   */
    p_grid_->setCurrentObject(dummyGrid);
    p_field_->setCurrentObject(dummyData);
    p_materials_->setCurrentObject(dummyMat);
}

int
ReadANSYS::compute(const char *)
{

    // cout << "ReadANSYS::compute called..." << endl;

    eqvStrainNotWritten = 0;
    int param;
    for (param = 0; param < hparams_.size(); ++param)
    {
        hparams_[param]->reset();
    }

    if (p_file_name_->isConnected())
    {
        std::string newFileName;
        if (extractName(newFileName) != 0)
        {
            sendError("Could not read file name from input port");
            return FAIL;
        }

        if (dynamic_cast<const coDoText *>(p_file_name_->getCurrentObject())->getTextLength() == 0
            || newFileName == "")
        {
            outputDummies();
            return SUCCESS;
        }
        if (newFileName != FileName_)
        {
            // do as when param is called reporting a name change
            // this includes testing if file may be successfully open
            FileName_ = newFileName;
        }

        if (fileNameChanged(0) != 0)
        {
            sendError("Could not read file as determined by input port");
            return FAIL;
        }

        useReadANSYSAttribute(p_file_name_->getCurrentObject());
        if (h_sol_->getIValue() == 2) // Nodal solution
        {
            open_err_ = SetNodeChoices();
            if (open_err_ != 0)
            {
                sendWarning("Problems when setting node choices");
                return FAIL;
            }
        }
    }

    if (open_err_ != 0)
    {
        sendError("Problem with results file");
        return FAIL;
    }
    // only geometry, nodal or element data?
    int problems = 0;
    switch (h_sol_->getIValue())
    {
    case 0:
        sendError("Please, choose some output type");
        outputDummies();
        break;
    case 1:
        problems = onlyGeometry();
        break;
    case 2:
        problems = nodalData();
        break;
    case 3:
        problems = derivedData();
        break;
    }
    readRST_.Reset(ReadRST::RADIKAL);
    if (problems)
        return FAIL;
/*
      if(   p_file_name_->isConnected()
         && p_grid_->getCurrentObject()){
         coFeedback feedback("SCA");
         addFeedbackParams(feedback,allFeedbackParams_);
         feedback.apply(p_grid_->getCurrentObject());
      }
   */
#ifdef _TEST_RST_FILE_NAME_
    coDoText *outname = new coDoText(p_outname_->getObjName(), FileName_.length() + 1);
    char *textname;
    outname->getAddress(&textname);
    strcpy(textname, FileName_.getValue());
    p_outname_->setCurrentObject(outname);
#endif
    return SUCCESS;

    new ANSYS();
}

int
ReadANSYS::getNumberOfNodes(int elem, int routine)
{
    ANSYS &elem_db_ = ANSYS::get_handle();

    int noCovNodes;

    switch (elem_db_.getCovType(routine))
    {
    case ANSYS::TYPE_TARGET:
    case ANSYS::TYPE_TARGET_2D:
    {
        // The implementation of TYPE_TARGET is not as trivial as it was done here.  Among the elements
        // that belong to this class are points, lines, parabolas, spheres, triangles, quadrilaterals,
        // 6-node triangles, 8-node quadrilaterals.  Besides is it meaningful to visualize contact
        // elements?  For the moment we comment this out.

        noCovNodes = 0;
        //          int node;
        //          const EType *etype = &readRST_.getETypes()[readRST_.getElements()[elem].type_-1];
        //          for(node=0;node<etype->nodes_;++node)
        //          {
        //             if(readRST_.getElements()[elem].nodes_[node] != 0)
        //             {
        //                ++noCovNodes;
        //             }
        //             else
        //             {
        //                break;
        //             }
        //          }
        //          if(elem_db_.getCovType(routine) == ANSYS::TYPE_TARGET && noCovNodes >= 6)
        //          {
        //             noCovNodes /= 2;
        //          }
        //          if(   elem_db_.getCovType(routine) == ANSYS::TYPE_TARGET_2D && noCovNodes > 2)
        //          {
        //             noCovNodes = 2;
        //          }
    }
    break;

    case ANSYS::TYPE_4_NODE_PLANE:
    {
        /****************************************************************************/
        /* 4-Node Planar Solids are rectangular or, in the degenerate case, triangular */
        /****************************************************************************/

        noCovNodes = 0;
        int node;
        int doublenodes = 0;
        int maxnodes = ANSYS::TYPE_4_NODE_PLANE + 8; // Total number of nodes including degenerate ones

        for (node = 0; node < maxnodes; ++node)
        {
            if (readRST_.getElements()[elem].nodes_[node] != 0)
            {
                for (int i = 0; i < node; i++)
                {
                    if (readRST_.getElements()[elem].nodes_[node] == readRST_.getElements()[elem].nodes_[i])
                    {
                        doublenodes++;
                        break;
                    }
                }
            }
            else
            {
                break;
            }
        }

        // TRIANGLE
        if (doublenodes == 1)
        {
            noCovNodes = 3;
        }

        // QUADRANGLE
        else if (doublenodes == 0)
        {
            noCovNodes = 4;
        }
    }
    break;

    case ANSYS::TYPE_8_NODE_PLANE:
    {
        /****************************************************************************/
        /* 8-Node Planar Solids are rectangular or, in the degenerate case, triangular */
        /****************************************************************************/

        noCovNodes = 0;

        const EType *etype = &readRST_.getETypes()[readRST_.getElements()[elem].type_ - 1];

        int keyopt1 = etype->keyops_[0]; // We assume here keyops[0] corresponds to KEYOPT(1)

        // Special case:  PLANE183
        if (routine == 183 && keyopt1 == 1)
        {
            noCovNodes = 3;
        }

        else
        {
            int node;
            int doublenodes = 0;
            int maxnodes = ANSYS::TYPE_8_NODE_PLANE; // Total number of nodes including degenerate ones

            for (node = 0; node < maxnodes; ++node)
            {
                if (readRST_.getElements()[elem].nodes_[node] != 0)
                {
                    for (int i = 0; i < node; i++)
                    {
                        if (readRST_.getElements()[elem].nodes_[node] == readRST_.getElements()[elem].nodes_[i])
                        {
                            doublenodes++;
                            break;
                        }
                    }
                }
                else
                {
                    break;
                }
            }

            // TRIANGLE
            if (doublenodes == 2)
            {
                noCovNodes = 3;
            }

            // QUADRANGLE
            else if (doublenodes == 0)
            {
                noCovNodes = 4;
            }
        }
    }
    break;

    case ANSYS::TYPE_10_NODE_SOLID:
    {
        /*****************************************************************/
        /* 10-Node Solids are always tetrahedral in shape, i.e. they do not */
        /* have degenerate derived elements                                              */
        /*****************************************************************/

        noCovNodes = 4;
    }
    break;

    case ANSYS::TYPE_20_NODE_SOLID:
    {
        int node;
        noCovNodes = 0;
        int doublenodes = 0;
        int maxnodes = ANSYS::TYPE_20_NODE_SOLID; // Total number of nodes including degenerate ones

        for (node = 0; node < maxnodes; ++node)
        {
            if (readRST_.getElements()[elem].nodes_[node] != 0)
            {
                for (int i = 0; i < node; i++)
                {
                    if (readRST_.getElements()[elem].nodes_[node] == readRST_.getElements()[elem].nodes_[i])
                    {
                        doublenodes++;
                        break;
                    }
                }
            }
            else
            {
                break;
            }
        }

        /***************************************************************/
        /* Determine correct element representation for 20-Node solids */
        /***************************************************************/

        // TETRAHEDRON
        if (doublenodes == 10)
        {
            noCovNodes = 4;
        }

        // PYRAMID
        else if (doublenodes == 7)
        {
            noCovNodes = 5;
        }

        // PRISM
        else if (doublenodes == 5)
        {
            noCovNodes = 6;
        }

        // HEXAHEDRON
        else if (doublenodes == 0)
        {
            noCovNodes = 8;
        }
    }
    break;

    case TYPE_HEXAEDER:
    {
        /*****************************************************************************************/
        /* 8-Node Solids are hexahedra or, in the degenerate case, prisms pyramids or tetrahedra */
        /*****************************************************************************************/

        noCovNodes = 0;
        int node;
        int doublenodes = 0;
        int maxnodes = UnstructuredGrid_Num_Nodes[elem_db_.getCovType(routine)]; // number of nodes if element is not degenerated

        for (node = 0; node < maxnodes; ++node)
        {
            if (readRST_.getElements()[elem].nodes_[node] != 0)
            {
                for (int i = 0; i < node; i++)
                {
                    if (readRST_.getElements()[elem].nodes_[node] == readRST_.getElements()[elem].nodes_[i])
                    {
                        doublenodes++;
                        break;
                    }
                }
            }
            else
            {
                break;
            }
        }

        // TETRAHEDRON
        if (doublenodes == 4)
        {
            noCovNodes = 4;
        }

        // PYRAMID
        else if (doublenodes == 3)
        {
            noCovNodes = 5;
        }

        // PRISM
        else if (doublenodes == 2)
        {
            noCovNodes = 6;
        }

        // HEXAHEDRON
        else if (doublenodes == 0)
        {
            noCovNodes = 8;
        }
    }
    break;

    // Why easy when you can do it complicated?
    //       case TYPE_HEXAEDER:
    //       {
    //          int node;
    //          noCovNodes=0;
    //          int doublenodes=0;
    //          //const EType *etype = &readRST_.getETypes()[readRST_.getElements()[elem].type_-1];
    //
    //          int maxnodes = UnstructuredGrid_Num_Nodes[elem_db_.getCovType(routine)]; // number of nodes if element is not degenerated
    //
    //          for(node=0;node<maxnodes;++node)
    //          {
    //             if(readRST_.getElements()[elem].nodes_[node] != 0)
    //             {
    //                for (int i=0;i<node;i++)
    //                {
    //                   if (readRST_.getElements()[elem].nodes_[node]==readRST_.getElements()[elem].nodes_[i])
    //                   {
    //                      doublenodes++;
    //                      break;
    //                   }
    //                }
    //             }
    //             else
    //             {
    //                break;
    //             }
    //          }
    //             if(doublenodes > 1)
    //             {
    //                 noCovNodes = UnstructuredGrid_Num_Nodes[elem_db_.getCovType(routine)]-(doublenodes);
    //                 // sort nodes
    //                 if(noCovNodes== 6) // prism
    //                 {
    //                       if (readRST_.getElements()[elem].nodes_[0]==readRST_.getElements()[elem].nodes_[1] &&
    //                       readRST_.getElements()[elem].nodes_[4]==readRST_.getElements()[elem].nodes_[5])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[1];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[2];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[7];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[1]==readRST_.getElements()[elem].nodes_[2]&&
    //                       readRST_.getElements()[elem].nodes_[5]==readRST_.getElements()[elem].nodes_[6])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[1];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[4];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[2]==readRST_.getElements()[elem].nodes_[3]&&
    //                       readRST_.getElements()[elem].nodes_[6]==readRST_.getElements()[elem].nodes_[7])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[1];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[4];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[7];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[3]==readRST_.getElements()[elem].nodes_[0]&&
    //                       readRST_.getElements()[elem].nodes_[7]==readRST_.getElements()[elem].nodes_[4])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[1];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[2];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[7];
    //                       }
    //
    //                       else if (readRST_.getElements()[elem].nodes_[2]==readRST_.getElements()[elem].nodes_[6] &&
    //                       readRST_.getElements()[elem].nodes_[1]==readRST_.getElements()[elem].nodes_[5])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[4];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[6]==readRST_.getElements()[elem].nodes_[7]&&
    //                       readRST_.getElements()[elem].nodes_[5]==readRST_.getElements()[elem].nodes_[4])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[2];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[4];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[1];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[7]==readRST_.getElements()[elem].nodes_[3]&&
    //                       readRST_.getElements()[elem].nodes_[4]==readRST_.getElements()[elem].nodes_[0])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[2];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[1];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[5];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[3]==readRST_.getElements()[elem].nodes_[2]&&
    //                       readRST_.getElements()[elem].nodes_[0]==readRST_.getElements()[elem].nodes_[1])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[4];
    //                       }
    //
    //
    //                       else if (readRST_.getElements()[elem].nodes_[0]==readRST_.getElements()[elem].nodes_[3]&&
    //                       readRST_.getElements()[elem].nodes_[1]==readRST_.getElements()[elem].nodes_[2])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[4];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[2];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[5];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[3]==readRST_.getElements()[elem].nodes_[7]&&
    //                       readRST_.getElements()[elem].nodes_[2]==readRST_.getElements()[elem].nodes_[6])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[4];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[1];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[7]==readRST_.getElements()[elem].nodes_[4]&&
    //                       readRST_.getElements()[elem].nodes_[6]==readRST_.getElements()[elem].nodes_[5])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[4];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[0];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[5];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[1];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[2];
    //                       }
    //                       else if (readRST_.getElements()[elem].nodes_[4]==readRST_.getElements()[elem].nodes_[0]&&
    //                       readRST_.getElements()[elem].nodes_[5]==readRST_.getElements()[elem].nodes_[1])
    //                       {
    //                          readRST_.getElements()[elem].nodes_[0]=readRST_.getElements()[elem].nodes_[3];
    //                          readRST_.getElements()[elem].nodes_[1]=readRST_.getElements()[elem].nodes_[7];
    //                          readRST_.getElements()[elem].nodes_[2]=readRST_.getElements()[elem].nodes_[4];
    //                          readRST_.getElements()[elem].nodes_[3]=readRST_.getElements()[elem].nodes_[2];
    //                          readRST_.getElements()[elem].nodes_[4]=readRST_.getElements()[elem].nodes_[6];
    //                          readRST_.getElements()[elem].nodes_[5]=readRST_.getElements()[elem].nodes_[5];
    //                       }
    //                 }
    //             }
    //             else
    //             {
    //                 noCovNodes = UnstructuredGrid_Num_Nodes[elem_db_.getCovType(routine)];
    //             }
    //       }
    //       break;

    default:
        noCovNodes = UnstructuredGrid_Num_Nodes[elem_db_.getCovType(routine)];
        break;
    }

    return noCovNodes;
}

int
ReadANSYS::onlyGeometry()
{
    ANSYS &elem_db_ = ANSYS::get_handle();

    std::vector<int> dummy;
    int problems = readRST_.Read(FileName_.c_str(), 1, dummy);
    if (problems)
        return problems;
    // now use nodeindex_, elemindex_, ety_, node_, element_
    int numVertices = 0;
    int elem;
    std::vector<int> e_l;
    std::vector<int> v_ansys_l;
    std::vector<int> t_l;
    int num_supp_elems = 0;
    // int nnodes;
    // int pos;
    // int doublenodes;

    for (elem = 0; elem < readRST_.getNumElement(); ++elem)
    {
        // for each element get the type of element...
        const EType *etype = &readRST_.getETypes()[readRST_.getElements()[elem].type_ - 1];
        int routine = etype->routine_;

        // CovType, ANSYSNodes, ... -> element library description

        int noCovNodes = getNumberOfNodes(elem, routine);

        if (noCovNodes <= 0)
            continue; // non-supported element
        ++num_supp_elems;

        t_l.push_back(elem_db_.ElementType(routine, noCovNodes));
        e_l.push_back(numVertices);

        int vert;

        //       if (routine==95) // or for any other supported degenerated elements .. so far just Solid 95 - others should be easy to implement
        //       {
        //          // we have to check for duplicated nodes here - we take the first noCovNodes that are different from each other
        //          nnodes=0;
        //          pos=0;
        //          doublenodes=0;
        //          do
        //          {
        //             doublenodes=0;
        //             for (int i=0;i<pos;i++)
        //             {
        //                if (readRST_.getElements()[elem].nodes_[pos]==readRST_.getElements()[elem].nodes_[i])
        //                {
        //                   doublenodes++;
        //                   break;
        //                }
        //             }
        //             if (!doublenodes)
        //             {
        //                v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[pos]);
        //                nnodes++;
        //             }
        //             pos++;
        //          }
        //          while (nnodes<noCovNodes);
        //       }

        /********************/
        /* PLANAR SOLIDS */
        /********************/

        // if(etype->nodes_ == ANSYS::TYPE_4_NODE_PLANE || etype->nodes_ == ANSYS::TYPE_8_NODE_PLANE)
        if (elem_db_.getCovType(routine) == ANSYS::TYPE_4_NODE_PLANE || elem_db_.getCovType(routine) == ANSYS::TYPE_8_NODE_PLANE)
        {
            switch (noCovNodes)
            {
            // TRIANGLE
            case 3:
            {
                for (vert = 0; vert < 3; ++vert)
                {
                    v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                }
            }
            break;

            // SQUARE
            case 4:
            {
                for (vert = 0; vert < 4; ++vert)
                {
                    v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                }
            }
            break;
            }
        }

        /*********************/
        /* 10-NODE SOLIDS */
        /*********************/

        // else if(etype->nodes_ == ANSYS::TYPE_10_NODE_SOLID)
        else if (elem_db_.getCovType(routine) == ANSYS::TYPE_10_NODE_SOLID)

        {
            // TETRAHEDRON
            for (vert = 0; vert < noCovNodes; ++vert)
            {
                v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
            }
        }

        /**********************************/
        /* 8-NODE AND 20-NODE SOLIDS */
        /**********************************/

        // else if(etype->nodes_ == TYPE_HEXAEDER || etype->nodes_ == ANSYS::TYPE_20_NODE_SOLID)
        else if (elem_db_.getCovType(routine) == TYPE_HEXAEDER || elem_db_.getCovType(routine) == ANSYS::TYPE_20_NODE_SOLID)
        {
            switch (noCovNodes)
            {
            // TETRAHEDRON
            case 4:
            {
                for (vert = 0; vert < 4; ++vert)
                {
                    if (vert != 3)
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                    }
                    else
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert + 1]);
                    }
                }
            }
            break;

            // PYRAMID
            case 5:
            {
                for (vert = 0; vert < 5; ++vert)
                {
                    v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                }
            }
            break;

            // PRISM
            case 6:
            {
                for (vert = 0; vert < 7; ++vert)
                {
                    if (vert != 3)
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                    }
                }
            }
            break;

            // HEXAHEDRON
            case 8:
            {
                for (vert = 0; vert < 8; ++vert)
                {
                    v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                }
            }
            break;
            }
        }

        else // take all nodes
        {
            for (vert = 0; vert < noCovNodes; ++vert)
            {
                v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
            }
        }
        numVertices += noCovNodes;
    }
    // decode nodes
    std::vector<int> nodeCodes;
    std::vector<float> x_l, y_l, z_l;
    int node;
    for (node = 0; node < readRST_.getNumNodes(); ++node)
    {
        nodeCodes.push_back(int(readRST_.getNodes()[node].id_));

        x_l.push_back(float(readRST_.getNodes()[node].x_));
        y_l.push_back(float(readRST_.getNodes()[node].y_));
        z_l.push_back(float(readRST_.getNodes()[node].z_));
    }
    Map1D nodeDecode(readRST_.getNumNodes(), &nodeCodes[0]);
    int vert;
    std::vector<int> v_l;
    for (vert = 0; vert < v_ansys_l.size(); ++vert)
    {
        v_l.push_back(nodeDecode[v_ansys_l[vert]]);
    }
    // make grid
    coDoUnstructuredGrid *entityGrid = new coDoUnstructuredGrid(p_grid_->getObjName(),
                                                                (int)e_l.size(), (int)v_l.size(), readRST_.getNumNodes(),
                                                                &e_l[0], &v_l[0],
                                                                &x_l[0], &y_l[0], &z_l[0],
                                                                &t_l[0]);
    entityGrid->addAttribute("COLOR", "White");
    p_grid_->setCurrentObject(entityGrid);
    return 0;
}

int
ReadANSYS::nodalData()
{
    ANSYS &elem_db_ = ANSYS::get_handle();
    if (h_nsol_->getIValue() == 0)
    {
        sendError("If you choose DOF data, then select a valid option");
        return -1;
    }
    int CovTime;
    std::vector<coDistributedObject *> grid_set_list;
    std::vector<coDistributedObject *> data_set_list;
    std::vector<coDistributedObject *> mat_set_list;

    readRST_.OpenFile(FileName_);
    int numTimeSteps = readRST_.getNumTimeSteps();

    for (CovTime = h_times_->getIValue(0) - 1;
         CovTime < h_times_->getIValue(1) && CovTime < numTimeSteps;
         CovTime += h_times_->getIValue(2))
    {
        // read raw data
        int problems = readRST_.Read(FileName_, CovTime + 1,
                                     DOFOptions_.codes_[h_nsol_->getIValue()]);
        if (problems)
            return problems;

        // realtime attribute
        char realTime[128];
        sprintf(realTime, "%g", readRST_.GetTime(CovTime));

        // now use nodeindex_, elemindex_, ety_, node_, element_
        int numVertices = 0;
        int elem;
        std::vector<int> e_l;
        std::vector<int> v_ansys_l;
        std::vector<int> t_l;
        int num_supp_elems = 0;
        // int nnodes;
        // int pos;
        // int doublenodes;

        for (elem = 0; elem < readRST_.getNumElement(); ++elem)
        {
            // for each element get the type of element...
            const EType *etype = &readRST_.getETypes()[readRST_.getElements()[elem].type_ - 1];
            int routine = etype->routine_;
            // CovType, ANSYSNodes, ... -> element library description
            int noCovNodes = getNumberOfNodes(elem, routine);
            if (noCovNodes <= 0)
                continue; // non-supported element
            ++num_supp_elems;
            t_l.push_back(elem_db_.ElementType(routine, noCovNodes));
            e_l.push_back(numVertices);
            int vert;

            //          if (routine==95) // or for any other supported degenerated elements .. so far just Solid 95 - others should be easy to implement
            //          {
            //             // we have to check for duplicated nodes here - we take the first noCovNodes that are different from each other
            //             nnodes=0;
            //             pos=0;
            //             doublenodes=0;
            //             do
            //             {
            //                doublenodes=0;
            //                for (int i=0;i<pos;i++)
            //                {
            //                   if (readRST_.getElements()[elem].nodes_[pos]==readRST_.getElements()[elem].nodes_[i])
            //                   {
            //                      doublenodes++;
            //                      break;
            //                   }
            //                }
            //                if (!doublenodes)
            //                {
            //                   v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[pos]);
            //                   nnodes++;
            //                }
            //                pos++;
            //             }
            //             while (nnodes<noCovNodes);
            //          }

            /********************/
            /* PLANAR SOLIDS */
            /********************/

            // if(etype->nodes_ == ANSYS::TYPE_4_NODE_PLANE || etype->nodes_ == ANSYS::TYPE_8_NODE_PLANE)
            if (elem_db_.getCovType(routine) == ANSYS::TYPE_4_NODE_PLANE || elem_db_.getCovType(routine) == ANSYS::TYPE_8_NODE_PLANE)
            {
                switch (noCovNodes)
                {
                // TRIANGLE
                case 3:
                {
                    for (vert = 0; vert < 3; ++vert)
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                    }
                }
                break;

                // SQUARE
                case 4:
                {
                    for (vert = 0; vert < 4; ++vert)
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                    }
                }
                break;
                }
            }

            /*********************/
            /* 10-NODE SOLIDS */
            /*********************/

            // else if(etype->nodes_ == ANSYS::TYPE_10_NODE_SOLID)
            else if (elem_db_.getCovType(routine) == ANSYS::TYPE_10_NODE_SOLID)
            {
                // TETRAHEDRON
                for (vert = 0; vert < noCovNodes; ++vert)
                {
                    v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                }
            }

            /**********************************/
            /* 8-NODE AND 20-NODE SOLIDS */
            /**********************************/

            // else if(etype->nodes_ == TYPE_HEXAEDER || etype->nodes_ == ANSYS::TYPE_20_NODE_SOLID)
            else if (elem_db_.getCovType(routine) == TYPE_HEXAEDER || elem_db_.getCovType(routine) == ANSYS::TYPE_20_NODE_SOLID)
            {
                switch (noCovNodes)
                {
                // TETRAHEDRON
                case 4:
                {
                    for (vert = 0; vert < 4; ++vert)
                    {
                        if (vert != 3)
                        {
                            v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                        }
                        else
                        {
                            v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert + 1]);
                        }
                    }
                }
                break;

                // PYRAMID
                case 5:
                {
                    for (vert = 0; vert < 5; ++vert)
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                    }
                }
                break;

                // PRISM
                case 6:
                {
                    for (vert = 0; vert < 7; ++vert)
                    {
                        if (vert != 3)
                        {
                            v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                        }
                    }
                }
                break;

                // HEXAHEDRON
                case 8:
                {
                    for (vert = 0; vert < 8; ++vert)
                    {
                        v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                    }
                }
                break;
                }
            }

            else // take all nodes
            {
                for (vert = 0; vert < noCovNodes; ++vert)
                {
                    v_ansys_l.push_back(readRST_.getElements()[elem].nodes_[vert]);
                }
            }
            numVertices += noCovNodes;
        }

        // decode nodes
        std::vector<int> nodeCodes;
        std::vector<float> x_l, y_l, z_l;
        int node;
        for (node = 0; node < readRST_.getNumNodes(); ++node)
        {
            nodeCodes.push_back(int(readRST_.getNodes()[node].id_));
            x_l.push_back(float(readRST_.getNodes()[node].x_));
            y_l.push_back(float(readRST_.getNodes()[node].y_));
            z_l.push_back(float(readRST_.getNodes()[node].z_));
        }
        Map1D nodeDecode(readRST_.getNumNodes(), &nodeCodes[0]);
        int vert;
        std::vector<int> v_l;
        for (vert = 0; vert < v_ansys_l.size(); ++vert)
        {
            v_l.push_back(nodeDecode[v_ansys_l[vert]]);
        }

        // make grid name
        char buf[64];
        sprintf(buf, "_%d", CovTime);
        std::string gridName(p_grid_->getObjName());
        gridName += buf;
        // find requested DOF AQUI
        // number of fields involved...
        size_t numOfFields = DOFOptions_.codes_[h_nsol_->getIValue()].size();
        std::string fieldName(p_field_->getObjName());
        std::string materialName(p_materials_->getObjName());
        fieldName += buf;
        materialName += buf;

        // gather materials in materials;
        int *materials = new int[readRST_.getNumElement()];
        for (elem = 0; elem < readRST_.getNumElement(); ++elem)
        {
            materials[elem] = readRST_.getElements()[elem].material_;
        }

        ReadDisplacements(nodeDecode);
        AddDisplacements(x_l, y_l, z_l);

        int fillField;
        // gather data in field[]
        float *field[3];
        switch (numOfFields)
        {
        case 1:
            //         field[0] = new float[readRST_.solheader_.numnodes_];
            field[0] = new float[x_l.size()];
            //         for(fillField = 0;fillField < readRST_.solheader_.numnodes_;
            for (fillField = 0; fillField < x_l.size();
                 ++fillField)
            {
                field[0][fillField] = ReadRST::FImpossible_;
            }
            for (node = 0; node < readRST_.DOFData_->nodesdataanz_; ++node)
            {
                // look for node ANSYS-code:
                int ANSYScode;
                // FIXME !!!!!!!!
                if (readRST_.DOFData_->nodesdataanz_
                    >= readRST_.solheader_.numnodes_
                    && node < readRST_.solheader_.numnodes_)
                {
                    ANSYScode = readRST_.getNodeIndex()[node];
                }
                else if (readRST_.DOFData_->nodesdataanz_
                         < readRST_.solheader_.numnodes_)
                {
                    ANSYScode = readRST_.getNodeIndex()[readRST_.DOFData_->nodesdata_[node] - 1];
                }
                else
                {
                    continue;
                }
                int cov_index = nodeDecode[ANSYScode];
                //            int cov_index = nodeDecode[readRST_.getNodeIndex()[node]];
                field[0][cov_index] = float(readRST_.DOFData_->data_[node]);
            }

            // cout << "output:" << endl;
            // cout << "element size = " << e_l.size() << endl;
            // cout << "connectivities = " << v_l.size() << endl;
            // cout << "nodes = " << x_l.size() << endl;
            // cout << "data = " << x_l.size() << endl;

            if (field[0])
            {
                MakeGridAndObjects(gridName, e_l, v_l, x_l, y_l, z_l, t_l,
                                   fieldName, &field[0], SCALAR,
                                   materialName, materials,
                                   grid_set_list, data_set_list, mat_set_list);
                delete[] field[0];
            }
            else
            {
                sendError("No data could be retrieved!");
            }

            break;

        case 3:
        {
            //         int num_nodes = readRST_.DOFData_->anz_/numOfFields;
            /*
                     field[0] = new float[readRST_.solheader_.numnodes_];
                     field[1] = new float[readRST_.solheader_.numnodes_];
                     field[2] = new float[readRST_.solheader_.numnodes_];
            */
            field[0] = new float[x_l.size()];
            field[1] = new float[x_l.size()];
            field[2] = new float[x_l.size()];
            //         for(fillField = 0;fillField < readRST_.solheader_.numnodes_;
            for (fillField = 0; fillField < x_l.size();
                 ++fillField)
            {
                field[0][fillField] = ReadRST::FImpossible_;
                field[1][fillField] = ReadRST::FImpossible_;
                field[2][fillField] = ReadRST::FImpossible_;
            }
            for (node = 0; node < readRST_.DOFData_->nodesdataanz_; ++node)
            {
                // look for node ANSYS-code:
                int ANSYScode;
                if (readRST_.DOFData_->nodesdataanz_
                    == readRST_.solheader_.numnodes_)
                {
                    ANSYScode = readRST_.getNodeIndex()[node];
                }
                else
                {
                    ANSYScode = readRST_.getNodeIndex()[readRST_.DOFData_->nodesdata_[node] - 1];
                }
                int cov_index = nodeDecode[ANSYScode];
                float vector[3];
                vector[0] = float(readRST_.DOFData_->data_[node]);
                vector[1] = float(readRST_.DOFData_->data_[node + readRST_.DOFData_->nodesdataanz_]);
                vector[2] = float(readRST_.DOFData_->data_[node + 2 * readRST_.DOFData_->nodesdataanz_]);
                const Rotation *rot = &readRST_.getNodes()[cov_index].Rotation_;
                field[0][cov_index] = rot->operator[](Rotation::XX) * vector[0] + rot->operator[](Rotation::XY) * vector[1] + rot->operator[](Rotation::XZ) * vector[2];
                field[1][cov_index] = rot->operator[](Rotation::YX) * vector[0] + rot->operator[](Rotation::YY) * vector[1] + rot->operator[](Rotation::YZ) * vector[2];
                field[2][cov_index] = rot->operator[](Rotation::ZX) * vector[0] + rot->operator[](Rotation::ZY) * vector[1] + rot->operator[](Rotation::ZZ) * vector[2];
            }

            if (field[0])
            {
                MakeGridAndObjects(gridName, e_l, v_l, x_l, y_l, z_l, t_l,
                                   fieldName, &field[0], VECTOR,
                                   materialName, materials,
                                   grid_set_list, data_set_list, mat_set_list);
                delete[] field[0];
                delete[] field[1];
                delete[] field[2];
            }
            else
            {
                sendError("No data could be retrieved!");
            }
        }
        break;
        default:
            sendError("The magnitude being processed is neither scalar nor vector, an unexpected problem has appeared");
            return -1;
        }
        delete[] materials;
        if (grid_set_list.size())
        {
            grid_set_list[grid_set_list.size() - 1]->addAttribute("REALTIME", realTime);
        }
        if (data_set_list.size())
        {
            data_set_list[data_set_list.size() - 1]->addAttribute("REALTIME", realTime);
        }
    } // end loop over CovTime
    grid_set_list.push_back(NULL);
    data_set_list.push_back(NULL);
    mat_set_list.push_back(NULL);

    coDoSet *gridOut = new coDoSet(p_grid_->getObjName(), &grid_set_list[0]);
    coDoSet *fieldOut = new coDoSet(p_field_->getObjName(), &data_set_list[0]);
    if ((grid_set_list.size() - 1) > 1)
    {
        ostringstream TimeSteps;
        TimeSteps << "1 " << grid_set_list.size() - 1 << endl;
        string timeSteps(TimeSteps.str());
        gridOut->addAttribute("TIMESTEP", timeSteps.c_str());
    }
    if ((data_set_list.size() - 1) > 1)
    {
        ostringstream TimeSteps;
        TimeSteps << "1 " << data_set_list.size() - 1 << endl;
        string timeSteps(TimeSteps.str());
        fieldOut->addAttribute("TIMESTEP", timeSteps.c_str());
    }
    p_grid_->setCurrentObject(gridOut);
    p_field_->setCurrentObject(fieldOut);
    p_materials_->setCurrentObject(new coDoSet(p_materials_->getObjName(), &mat_set_list[0]));
    return 0;
}

int
ReadANSYS::fileNameChanged(int yell)
{
    if (!p_file_name_->isConnected())
    {
        FileName_ = p_rst_->getValue();
    }
    readRST_.Reset(ReadRST::RADIKAL);
    open_err_ = readRST_.OpenFile(FileName_.c_str());
    switch (open_err_)
    {
    case 0: // wonderful
        break;
    case 1:
        if (yell)
        {
            sendError("Could not open results file");
        }
        else
        {
            sendWarning("Could not open results file");
        }
        break;
    case 2:
        if (yell)
        {
            sendError("Could not read header");
        }
        else
        {
            sendWarning("Could not read header");
        }
        break;
    case 3:
        if (yell)
        {
            sendError("Could not read nodal equivalence table");
        }
        else
        {
            sendWarning("Could not read nodal equivalence table");
        }
        break;
    case 4:
        if (yell)
        {
            sendError("Could not read element equivalence table");
        }
        else
        {
            sendWarning("Could not read element equivalence table");
        }
        break;
    case 5:
        if (yell)
        {
            sendError("Could not read time table");
        }
        else
        {
            sendWarning("Could not read time table");
        }
        break;
    default:
        if (yell)
        {
            sendError("ReadRST::OpenFile returned unsupported value");
        }
        else
        {
            sendWarning("ReadRST::OpenFile returned unsupported value");
        }
        break;
    }

    if (open_err_ != 0)
        return -1;

    // get the number of time steps for the default last time step to be shown
    p_sol_->enable();
    if (!inMapLoading)
    {
        p_sol_->setValue(1);
    }
    p_sol_->show();
    p_times_->enable();
    if (!inMapLoading)
    {
        p_times_->setValue(0, 1);
        p_times_->setValue(1, readRST_.getNumTimeSteps());
        p_times_->setValue(2, 1);
    }
    p_times_->show();
    p_nsol_->hide();
    p_esol_->hide();
    p_stress_->hide();
    p_beam_stress_->hide();
    p_axi_shell_stress_->hide();
    p_thermalFlux_->hide();
    p_vol_energy_->hide();
    p_nsol_->disable();
    p_esol_->disable();
    p_stress_->disable();
    p_beam_stress_->disable();
    p_axi_shell_stress_->disable();
    p_thermalFlux_->disable();
    p_vol_energy_->disable();

    return 0;
}

void
ReadANSYS::param(const char *paramName, bool inMapLoading)
{
    this->inMapLoading = inMapLoading;
    if (p_file_name_->isConnected())
    {
        return;
    }
    if (strcmp(paramName, p_rst_->getName()) == 0
        /* && !p_file_name_->isConnected() */)
    {
        if (oldFileName && strcmp(oldFileName, p_rst_->getValue()) == 0) // only read file and reset parameters if filename actually changed
        {
        }
        else
        {
            delete[] oldFileName;
            oldFileName = new char[strlen(p_rst_->getValue()) + 1];
            strcpy(oldFileName, p_rst_->getValue());
            fileNameChanged(1);
        }
        return;
    }

    if (strcmp(paramName, p_sol_->getName()) == 0)
    {
        if (open_err_ != 0)
        {
            sendError("Problem with the results file");
            return;
        }
        switch (p_sol_->getValue())
        {
        case 0:
        case 1: // geometry
            p_nsol_->hide();
            p_nsol_->disable();
            p_esol_->hide();
            p_esol_->disable();
            p_output_node_decode_->disable();
            break;
        case 2: // DOFs
            p_esol_->hide();
            p_esol_->disable();
            p_nsol_->enable();
            open_err_ = SetNodeChoices();
            p_nsol_->show();
            p_output_node_decode_->disable();
            break;
        case 3: // Derived data
            p_nsol_->disable();
            p_nsol_->hide();
            p_esol_->enable();
            if (!inMapLoading)
            {
                p_esol_->setValue(0);
            }
            p_esol_->show();
            p_output_node_decode_->enable();
            break;
        }
        p_stress_->disable();
        p_stress_->hide();
        p_beam_stress_->disable();
        p_beam_stress_->hide();
        p_axi_shell_stress_->disable();
        p_axi_shell_stress_->hide();
        p_thermalFlux_->disable();
        p_thermalFlux_->hide();
        p_vol_energy_->disable();
        p_vol_energy_->hide();
        p_mag_flux_dens_->disable();
        p_mag_flux_dens_->hide();
    }

    if (strcmp(paramName, p_esol_->getName()) == 0)
    {
        if (open_err_ != 0)
        {
            sendError("There is still a problem with the results file");
            return;
        }
        if (p_esol_->getValue() == 0)
        {
            p_stress_->disable();
            p_stress_->hide();
            p_beam_stress_->disable();
            p_beam_stress_->hide();
            p_axi_shell_stress_->disable();
            p_axi_shell_stress_->hide();
            p_top_bottom_->disable();
            p_top_bottom_->hide();
            p_thermalFlux_->disable();
            p_thermalFlux_->hide();
            p_vol_energy_->disable();
            p_vol_energy_->hide();
            p_mag_flux_dens_->disable();
            p_mag_flux_dens_->hide();
        }
        else if (p_esol_->getValue() < 6) // all options but thermal fluxes or energies
        {
            p_stress_->enable();
            p_stress_->show();
            p_beam_stress_->enable();
            p_beam_stress_->show();
            p_top_bottom_->enable();
            p_top_bottom_->show();
            p_axi_shell_stress_->enable();
            p_axi_shell_stress_->show();
            p_thermalFlux_->disable();
            p_thermalFlux_->hide();
            p_vol_energy_->disable();
            p_vol_energy_->hide();
            p_mag_flux_dens_->disable();
            p_mag_flux_dens_->hide();
        }
        else if (p_esol_->getValue() == 6)
        {
            p_stress_->disable();
            p_stress_->hide();
            p_beam_stress_->disable();
            p_beam_stress_->hide();
            p_axi_shell_stress_->disable();
            p_axi_shell_stress_->hide();
            p_top_bottom_->disable();
            p_top_bottom_->hide();
            p_thermalFlux_->enable();
            p_thermalFlux_->show();
            p_vol_energy_->disable();
            p_vol_energy_->hide();
            p_mag_flux_dens_->disable();
            p_mag_flux_dens_->hide();
        }
        else if (p_esol_->getValue() == 7)
        {
            p_stress_->disable();
            p_stress_->hide();
            p_beam_stress_->disable();
            p_beam_stress_->hide();
            p_axi_shell_stress_->disable();
            p_axi_shell_stress_->hide();
            p_top_bottom_->disable();
            p_top_bottom_->hide();
            p_thermalFlux_->disable();
            p_thermalFlux_->hide();
            p_vol_energy_->enable();
            p_vol_energy_->show();
            p_mag_flux_dens_->disable();
            p_mag_flux_dens_->hide();
        }
        else if (p_esol_->getValue() == 8)
        {
            p_stress_->disable();
            p_stress_->hide();
            p_beam_stress_->disable();
            p_beam_stress_->hide();
            p_axi_shell_stress_->disable();
            p_axi_shell_stress_->hide();
            p_top_bottom_->disable();
            p_top_bottom_->hide();
            p_thermalFlux_->disable();
            p_thermalFlux_->hide();
            p_vol_energy_->disable();
            p_vol_energy_->hide();
            p_mag_flux_dens_->enable();
            p_mag_flux_dens_->show();
        }
    }
    this->inMapLoading = false;
}

// gather displacements
void
ReadANSYS::ReadDisplacements(const Map1D &nodeDecode)
{
    displacements_[0].clear();
    displacements_[1].clear();
    displacements_[2].clear();

    int fillField;
    //   for(fillField = 0;fillField < readRST_.solheader_.numnodes_;
    for (fillField = 0; fillField < readRST_.getNumNodes();
         ++fillField)
    {
        displacements_[0].push_back(0.0);
        displacements_[1].push_back(0.0);
        displacements_[2].push_back(0.0);
    }
    int node;
    for (node = 0; node < readRST_.DOFData_->nodesdataanz_; ++node)
    {
        // look for node ANSYS-code:
        int ANSYScode;
        /*
            if(1 ||   readRST_.DOFData_->nodesdataanz_ // FIXME !!!!!!!!!!!
               == readRST_.solheader_.numnodes_){
               ANSYScode = readRST_.getNodeIndex()[node];
            }
            else {
               ANSYScode = readRST_.getNodeIndex()[
                               readRST_.DOFData_->nodesdata_[node]-1];
            }
      */
        if (readRST_.DOFData_->nodesdataanz_ // FIXME !!!!!!!!
            >= readRST_.solheader_.numnodes_
            && node < readRST_.solheader_.numnodes_)
        {
            ANSYScode = readRST_.getNodeIndex()[node];
        }
        else if (readRST_.DOFData_->nodesdataanz_
                 < readRST_.solheader_.numnodes_)
        {
            ANSYScode = readRST_.getNodeIndex()[readRST_.DOFData_->nodesdata_[node] - 1];
        }
        else
        {
            continue;
        }
        int cov_index = nodeDecode[ANSYScode];
        displacements_[0][cov_index] = float(readRST_.DOFData_->displacements_[node]);
        displacements_[1][cov_index] = float(readRST_.DOFData_->displacements_[node + readRST_.DOFData_->nodesdataanz_]);
        displacements_[2][cov_index] = float(readRST_.DOFData_->displacements_[node + 2 * readRST_.DOFData_->nodesdataanz_]);
    }
}

void
ReadANSYS::AddDisplacements(std::vector<float> &x_l, std::vector<float> &y_l, std::vector<float> &z_l)
{
    size_t length = x_l.size();
    int node;
    float factor = h_scale_->getFValue();
    Node *pnode = const_cast<Node *>(readRST_.getNodes());
    for (node = 0; node < length; ++node, ++pnode)
    {
        Rotation *p_rot = &pnode->Rotation_;
        x_l[node] += factor * ((*p_rot)[Rotation::XX] * displacements_[0][node] + (*p_rot)[Rotation::XY] * displacements_[1][node] + (*p_rot)[Rotation::XZ] * displacements_[2][node]);
        y_l[node] += factor * ((*p_rot)[Rotation::YX] * displacements_[0][node] + (*p_rot)[Rotation::YY] * displacements_[1][node] + (*p_rot)[Rotation::YZ] * displacements_[2][node]);
        z_l[node] += factor * ((*p_rot)[Rotation::ZX] * displacements_[0][node] + (*p_rot)[Rotation::ZY] * displacements_[1][node] + (*p_rot)[Rotation::ZZ] * displacements_[2][node]);
    }
}

MODULE_MAIN(IO, ReadANSYS)
