/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadANSYS.h"
#include <util/coviseCompat.h>
#include "ANSYS.h"
#include <alg/coCellToVert.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>

#define _INLINE_
#ifndef __linux__
#define M_INLINE_ inline
#endif

_INLINE_ int
ReadANSYS::outputIsVector()
{

    return ((h_esol_->getIValue() == 6 && h_thermalFlux_->getIValue() == 4) || (h_esol_->getIValue() == 8 && h_mag_flux_dens_->getIValue() == 0));
}

int
ReadANSYS::derivedData()
{
    ANSYS &elem_db_ = ANSYS::get_handle();
    if (h_esol_->getIValue() == 0)
    {
        sendError("If you choose derived data, then select a valid option");
        return -1;
    }
    int thereAreUnsupportedEls = 0;
    int CovTime;
    std::vector<coDistributedObject *> grid_set_list;
    std::vector<coDistributedObject *> data_set_list;
    std::vector<coDistributedObject *> mat_set_list;
    std::vector<coDistributedObject *> node_decode_set_list;

    readRST_.OpenFile(FileName_);
    int numTimeSteps = readRST_.getNumTimeSteps();

    for (CovTime = h_times_->getIValue(0) - 1;
         CovTime < h_times_->getIValue(1) && CovTime < numTimeSteps;
         CovTime += h_times_->getIValue(2))
    {
        int problems;
        std::vector<int> dummy;
        problems = readRST_.Read(FileName_, CovTime + 1, dummy);
        if (problems)
            return problems;
        // read raw data
        switch (h_esol_->getIValue())
        {
        case 1:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::STRESS);
            break;
        case 2:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::E_EL);
            break;
        case 3:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::E_PLAS);
            break;
        case 4:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::E_CREEP);
            break;
        case 5:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::E_THERM);
            break;
        case 6:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::FIELD_FLUX);
            break;
        case 7:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::VOL_ENERGY);
            break;
        case 8:
            if (readRST_.getVersion() < 10)
                problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                                ReadRST::E_THERM);
            else
                problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                                ReadRST::M_FLUX_DENS);
            break;
        case 9:
            problems = readRST_.ReadDerived(FileName_, CovTime + 1,
                                            ReadRST::E_TEMP);
            break;
        default:
            sendError("Derived data: invalid option");
            return -1;
        }
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
        // int nnodes;
        // int pos;
        // int doublenodes;

        // cout << "Some statistics:" << endl;
        // cout << "Number of elements: " << readRST_.getNumElement() << endl;

        for (elem = 0; elem < readRST_.getNumElement(); ++elem)
        {
            // for each element get the type of element...
            const EType *etype = &readRST_.getETypes()[readRST_.getElements()[elem].type_ - 1];
            int routine = etype->routine_;

            // CovType, ANSYSNodes, ... -> element library description
            int noCovNodes = getNumberOfNodes(elem, routine);

            // if(noCovNodes<=0) continue; // non-supported element
            if (noCovNodes <= 0 && thereAreUnsupportedEls == 0)
            {
                sendWarning("There are unsupported elements, which will not be shown");
                thereAreUnsupportedEls = 1;
            }

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

            else if (etype->nodes_ == ANSYS::TYPE_10_NODE_SOLID)
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
        // map from ANSYS labels to "natural" labels
        Map1D nodeDecode(readRST_.getNumNodes(), &nodeCodes[0]);
        int vert;
        std::vector<int> v_l;
        for (vert = 0; vert < v_ansys_l.size(); ++vert)
        {
            v_l.push_back(nodeDecode[v_ansys_l[vert]]);
        }

        // decode Elems
        std::vector<int> elemCodes;
        for (elem = 0; elem < readRST_.getNumElement(); ++elem)
        {
            elemCodes.push_back(readRST_.getElements()[elem].num_);
        }
        Map1D elemDecode(readRST_.getNumElement(), &elemCodes[0]);

        // make names
        char buf[64];
        sprintf(buf, "_%d", CovTime);
        std::string gridName(p_grid_->getObjName());
        gridName += buf;
        std::string fieldName(p_field_->getObjName());
        std::string materialName(p_materials_->getObjName());
        fieldName += buf;
        materialName += buf;
        // FIXME port
        //      std::string nodeIndName(p_node_decode_->getObjName());
        std::string nodeIndName(p_field_->getObjName());
        nodeIndName += buf;

        // gather materials in materials;
        int *materials = new int[readRST_.getNumElement()];
        for (elem = 0; elem < readRST_.getNumElement(); ++elem)
        {
            materials[elem] = readRST_.getElements()[elem].material_;
        }
        ReadDisplacements(nodeDecode);
        AddDisplacements(x_l, y_l, z_l);

        NodeValues ansNodeValues;
        ElemValues ansElemValues;

        vector<float> tmp_vec;
        vector<float>::iterator vit;
        vector<float> vals;

        // cout << "readRST_.DerivedData_->anz_ = " << readRST_.DerivedData_->anz_ << endl;

        for (elem = 0; elem < readRST_.DerivedData_->anz_; ++elem)
        {
            // look for node ANSYS-code:
            int ANSYScode;
            ANSYScode = readRST_.getElemIndex()[elem];

            int cov_index = elemDecode[ANSYScode];
            // info for ProcessField: field, element routine, magnitude, component
            // for each element get the type of element...
            const EType *etype = &readRST_.getETypes()[readRST_.getElements()[cov_index].type_ - 1];
            int routine = etype->routine_;

            tmp_vec = ProcessDerivedField(readRST_.DerivedData_->data_[elem], routine);

            if ((tmp_vec.size() == 1 && tmp_vec[0] != ReadRST::FImpossible_) || (outputIsVector() && tmp_vec.size() == 3)) // data is per element
            {
                ansElemValues.push_back(tmp_vec);
            }
            else
            {
                int num_entries = outputIsVector() ? tmp_vec.size() / 3 : tmp_vec.size();
                for (int j = 0; j < (outputIsVector() ? 3 : 1); j++)
                {
                    for (int i = 0; i < num_entries; i++)
                    {
                        if (ansNodeValues.find(readRST_.getElements()[cov_index].nodes_[i]) != ansNodeValues.end() && ansNodeValues[readRST_.getElements()[cov_index].nodes_[i]][0][0] != ReadRST::FImpossible_)
                        {
                            ansNodeValues[readRST_.getElements()[cov_index].nodes_[i]][j].push_back(tmp_vec[i + j * num_entries]);
                        }
                        else
                        {
                            vector<vector<float> > vv;
                            vector<float> v, v1, v2;
                            v.push_back(tmp_vec[i]);
                            vv.push_back(v);
                            vv.push_back(v1);
                            vv.push_back(v2);
                            ansNodeValues[readRST_.getElements()[cov_index].nodes_[i]] = vv;
                        }
#ifdef DEBUG
                        if (elem < 10)
                            cout << "ANS " << ANSYScode << " : " << readRST_.getElements()[cov_index].nodes_[i] << " " << tmp_vec[i] << endl;
#endif
                    }
                }
            }
        }

        // cout << "tmp_vec = " << tmp_vec.size() << endl;
        // cout << "ansElemValues = " << ansElemValues.size() << endl;
        // cout << "ansNodeValues = " << ansNodeValues.size() << endl;
        bool data_ok;

        data_ok = true;

        // Average all values of one node
        AvgNodeValues covAvgValues;

        if (!ansNodeValues.empty())
        {
            float avg = 0.0;
            NodeValues::iterator it;

            for (it = ansNodeValues.begin(); it != ansNodeValues.end(); it++)
            {
                int cov_index = nodeDecode[(*it).first];
                vector<float> av;
                for (int j = 0; j < (outputIsVector() ? 3 : 1); j++)
                {
                    tmp_vec = (*it).second[j];
                    avg = 0.;
                    int num = 0;

                    if (tmp_vec.size() > 0)
                    {
                        for (vit = tmp_vec.begin(); vit != tmp_vec.end(); vit++)
                        {
                            if (*vit != ReadRST::FImpossible_)
                            {
                                avg += *vit;
                                num++;
                            }
                        }
                        if (num > 0)
                        {
                            avg /= num;
                        }
                        else
                        {
                            avg = ReadRST::FImpossible_;
                        }
                    }
                    else
                    {
                        avg = ReadRST::FImpossible_;
                    }
                    av.push_back(avg);
                }
                covAvgValues[cov_index] = av;
            }
        }

        // AQUI
        std::vector<float> f_l_c[3];
        if (!covAvgValues.empty())
        {
            fprintf(stderr, "!covAvgValues.empty()\n");
            //fprintf(stderr,"x_l.size()=%d\n",x_l.size());
            for (int i = 0; i < x_l.size(); i++)
            {
                if (covAvgValues.find(i) != covAvgValues.end())
                {
                    f_l_c[0].push_back(covAvgValues[i][0]);
                    if (outputIsVector())
                    {
                        f_l_c[1].push_back(covAvgValues[i][1]);
                        f_l_c[2].push_back(covAvgValues[i][2]);
                    }
                }
                else
                {
                    f_l_c[0].push_back(ReadRST::FImpossible_);
                    f_l_c[1].push_back(ReadRST::FImpossible_);
                    f_l_c[2].push_back(ReadRST::FImpossible_);
                }
            }
        }
        else if (!ansElemValues.empty() && !thereAreUnsupportedEls)
        {
            fprintf(stderr, "!ansElemValues.empty() && !thereAreUnsupportedEls\n");

            // cout << "output:" << endl;
            // cout << "element size = " << e_l.size() << endl;
            // cout << "connectivities = " << v_l.size() << endl;
            // cout << "nodes = " << x_l.size() << endl;
            // cout << "data = " << f_l_c[0].size() << endl;

            if (ansElemValues.size() == e_l.size())
            {
                for (int i = 0; i < e_l.size(); i++)
                {
                    f_l_c[0].push_back(ansElemValues[i][0]);
                    if (outputIsVector())
                    {
                        f_l_c[1].push_back(ansElemValues[i][1]);
                        f_l_c[2].push_back(ansElemValues[i][2]);
                    }
                }
            }
            else if (ansElemValues.size() == 0)
            {
                sendError("No element data could be retrieved!");
                data_ok = false;
            }
            else
            {
                sendError("A problem while reading data was detected!");
                data_ok = false;
            }
        }

        if (thereAreUnsupportedEls)
        {
            fprintf(stderr, "thereAreUnsupportedEls!\n");
            // eliminate non-supported elements
            std::vector<int> e_l_c;
            std::vector<int> t_l_c;

            if (ansElemValues.size() != 0 && ansElemValues.size() == e_l.size())
            {
                for (elem = 0; elem < readRST_.getNumElement(); ++elem)
                {
                    if (t_l[elem] != TYPE_NONE)
                    {
                        e_l_c.push_back(e_l[elem]);
                        t_l_c.push_back(t_l[elem]);
                        if (!ansElemValues.empty())
                        {
                            f_l_c[0].push_back(ansElemValues[elem][0]);
                            if (outputIsVector())
                            {
                                f_l_c[1].push_back(ansElemValues[elem][1]);
                                f_l_c[2].push_back(ansElemValues[elem][2]);
                            }
                        }
                    }
                }
            }
            else if (ansElemValues.size() == 0)
            {
                sendError("No element data could be retrieved!");
                data_ok = false;
            }
            else
            {
                sendError("A problem while reading data was detected!");
                data_ok = false;
            }

            if (!outputIsVector())
            {
                // cout << "output:" << endl;
                // cout << "element size = " << e_l_c.size() << endl;
                // cout << "connectivities = " << v_l.size() << endl;
                // cout << "nodes = " << x_l.size() << endl;
                // cout << "data = " << f_l_c[0].size() << endl;

                if (data_ok)
                {
                    MakeGridAndObjectsElems(gridName, e_l_c, v_l, x_l, y_l, z_l, t_l_c,
                                            fieldName, &f_l_c[0][0], NULL, NULL,
                                            materialName, materials, nodeIndName,
                                            grid_set_list, data_set_list, mat_set_list,
                                            node_decode_set_list, !ansElemValues.empty());
                }
            }
            else
            {
                // cout << "output:" << endl;
                // cout << "element size = " << e_l_c.size() << endl;
                // cout << "connectivities = " << v_l.size() << endl;
                // cout << "nodes = " << x_l.size() << endl;
                // cout << "data = " << f_l_c[0].size() << endl;

                if (data_ok)
                {

                    MakeGridAndObjectsElems(gridName, e_l_c, v_l, x_l, y_l, z_l, t_l_c,
                                            fieldName, &f_l_c[0][0],
                                            &f_l_c[1][0], &f_l_c[2][0],
                                            materialName, materials, nodeIndName,
                                            grid_set_list, data_set_list, mat_set_list,
                                            node_decode_set_list, !ansElemValues.empty());
                }
            }
        }
        else
        {
            // all elements are supported
            if (!outputIsVector())
            {
                // cout << "all elements are supported" << endl;
                // cout << "output:" << endl;
                // cout << "element size = " << e_l.size() << endl;
                // cout << "connectivities = " << v_l.size() << endl;
                // cout << "nodes = " << x_l.size() << endl;
                // cout << "data = " << f_l_c[0].size() << endl;

                MakeGridAndObjectsElems(gridName, e_l, v_l, x_l, y_l, z_l, t_l,
                                        fieldName, &f_l_c[0][0], NULL, NULL,
                                        materialName, materials, nodeIndName,
                                        grid_set_list, data_set_list, mat_set_list,
                                        node_decode_set_list, !ansElemValues.empty());
            }
            else
            {
                // cout << "all elements are supported" << endl;
                // cout << "output:" << endl;
                // cout << "element size = " << e_l.size() << endl;
                // cout << "connectivities = " << v_l.size() << endl;
                // cout << "nodes = " << x_l.size() << endl;
                // cout << "data = " << f_l_c[0].size() << endl;

                MakeGridAndObjectsElems(gridName, e_l, v_l, x_l, y_l, z_l, t_l,
                                        fieldName, &f_l_c[0][0], &f_l_c[1][0], &f_l_c[2][0],
                                        materialName, materials, nodeIndName,
                                        grid_set_list, data_set_list, mat_set_list,
                                        node_decode_set_list, !ansElemValues.empty());
            }
        }

        if (grid_set_list.size())
        {
            grid_set_list[grid_set_list.size() - 1]->addAttribute("REALTIME", realTime);
        }
        if (data_set_list.size())
        {
            data_set_list[data_set_list.size() - 1]->addAttribute("REALTIME", realTime);
        }
    }
    grid_set_list.push_back(NULL);
    data_set_list.push_back(NULL);
    mat_set_list.push_back(NULL);

    coDoSet *gridOut = new coDoSet(p_grid_->getObjName(), &grid_set_list[0]);
    if ((grid_set_list.size() - 1) > 1)
    {
        ostringstream TimeSteps;
        TimeSteps << "1 " << grid_set_list.size() - 1 << endl;
        string timeSteps(TimeSteps.str());
        gridOut->addAttribute("TIMESTEP", timeSteps.c_str());
    }
    p_grid_->setCurrentObject(gridOut);
    p_materials_->setCurrentObject(new coDoSet(p_materials_->getObjName(), &mat_set_list[0]));

    if (h_output_node_decode_->getIValue())
    {
        node_decode_set_list.push_back(NULL);
        // FIXME port
        coDoSet *nodeDecodeOut = new coDoSet(p_field_->getObjName(),
                                             &node_decode_set_list[0]);
        // FIXME port p_node_decode_->setCurrentObject(nodeDecodeOut);
        p_field_->setCurrentObject(nodeDecodeOut);
    }
    else
    {
        coDoSet *fieldOut = new coDoSet(p_field_->getObjName(), &data_set_list[0]);
        if ((data_set_list.size() - 1) > 1)
        {
            ostringstream TimeSteps;
            TimeSteps << "1 " << data_set_list.size() - 1 << endl;
            string timeSteps(TimeSteps.str());
            fieldOut->addAttribute("TIMESTEP", timeSteps.c_str());
        }
        p_field_->setCurrentObject(fieldOut);
    }
    return 0;
}

vector<float>
ReadANSYS::ProcessDerivedField(std::vector<double> &derData,
                               int routine)
{
    ANSYS &elem_db_ = ANSYS::get_handle();

    vector<float> ret;
    if (derData.size() == 0
        || (derData.size() == 1
            && derData[0] == ReadRST::FImpossible_))
    {
        ret.push_back(ReadRST::FImpossible_);
        return ret;
    }
    int offset = 0;
    int component;
    int topBot = h_top_bottom_->getIValue();
    int numTimes = 0, node, period = 0;

    std::vector<double> enhData;
    std::vector<double> *ptrData = &derData;

    switch (h_esol_->getIValue())
    {
    case 1: // stresses
        component = h_stress_->getIValue() - 1;
        switch (elem_db_.getStressSupport(routine))
        {
        case ANSYS::SOLID:
        {
            if (component < 0)
                break;
            period = 11;
            numTimes = derData.size() / period;

            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
            break;
        }
        case ANSYS::SHELL:
        {
            if (component < 0)
                break;
            period = 22;
            numTimes = derData.size() / period;
            switch (topBot)
            {
            case BOTTOM:
                offset = derData.size() / 2;

            case TOP:
                for (node = 0; node < numTimes; ++node)
                {
                    ret.push_back(derData[node * period + component + offset]);
                }
                break;

            case AVERAGE:
                for (node = 0; node < numTimes; ++node)
                {
                    ret.push_back(.5 * (derData[node * period + component] + derData[node * period + component + derData.size() / 2]));
                }
                break;

            default:
                sendWarning("Unknonw value for topBot in ReadANSYS::ProcessDerivedField");
                ret.push_back(ReadRST::FImpossible_);
                break;
            }
            break;
        }

        case ANSYS::LINK:
            component = h_beam_stress_->getIValue() - 1;
            if (component != 0)
                break;
            ret.push_back(derData[0]);
            break;

        case ANSYS::BEAM3:
        case ANSYS::BEAM4:
            component = h_beam_stress_->getIValue() - 1;
            if (component < 0)
                break;
            if (elem_db_.getStressSupport(routine) == ANSYS::BEAM3)
            {
                if (component < 0 || component > 2)
                {
                    return ret;
                }
                else
                {
                    period = 3;
                }
            }
            else if (elem_db_.getStressSupport(routine) == ANSYS::BEAM4)
            {
                if (component < 0 || component > 4)
                {
                    return ret;
                }
                else
                {
                    period = 5;
                }
            }
            else
            {
                sendWarning("Unhandled value for StressSupport_[routine] in ReadANSYS::ProcessDerivedField");
                ret.push_back(ReadRST::FImpossible_);
                return ret;
            }

            numTimes = derData.size() / period;
            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
            break;

        case ANSYS::PLANE:
            if (component < 0)
                break;
            period = 11;
            numTimes = derData.size() / period;
            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
            break;

        case ANSYS::NO_STRESS:
            break;

        default:
            break;
        }
        break;

    case 2: // elastic strain
    case 3: // plastic strain
    case 4: // creep strain
    case 5: // thermal strain
        component = h_stress_->getIValue() - 1;
        switch (elem_db_.getStressSupport(routine))
        {
        case ANSYS::SOLID:
            if (component < 0)
                break;
            if (component > 5 && component != 10)
                break;
            if (component == 10)
                component = 6;
            period = 7;
            numTimes = derData.size() / period;
            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
            break;

        case ANSYS::LINK:
            component = h_beam_stress_->getIValue() - 1;
            if (component != 0)
                break;
            ret.push_back(derData[0]);
            break;

        case ANSYS::BEAM3:
        case ANSYS::BEAM4:
            component = h_beam_stress_->getIValue() - 1;
            if (component < 0)
                break;
            if (elem_db_.getStressSupport(routine) == ANSYS::BEAM3)
            {
                if (component < 0 || component > 2)
                {
                    return ret;
                }
                else
                {
                    period = 3;
                }
            }
            else if (elem_db_.getStressSupport(routine) == ANSYS::BEAM4)
            {
                if (component < 0 || component > 4)
                {
                    return ret;
                }
                else
                {
                    period = 5;
                }
            }
            else
            {
                sendWarning("Unhandled value for StressSupport_[routine] in ReadANSYS::ProcessDerivedField");
                ret.push_back(ReadRST::FImpossible_);
                return ret;
            }
            numTimes = derData.size() / period;
            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
            break;

        case ANSYS::PLANE:
            if (component < 0)
                break;
            period = 7;
            if (component > 5)
            {
                if (component != 10)
                {
                    return ret;
                }
                else
                {
                    component = 6;
                }
            }

            numTimes = derData.size() / period;

            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
            break;

        case ANSYS::AXI_SHELL:
            component = h_axi_shell_stress_->getIValue() - 1;
            if (component < 0)
                break;
            if (derData.size() != 12)
                break;
            switch (topBot)
            {
            case TOP:
                offset = 0;
                break;
            case AVERAGE:
                offset = 4;
                break;
            case BOTTOM:
                offset = 8;
                break;
            }
            ret.push_back(derData[component + offset]);
            break;

        case ANSYS::SHELL:
            if (component < 0)
                break;
            if (component > 5)
            {
                if (component != 10)
                {
                    return ret;
                }
                else
                {
                    component = 6;
                }
            }
            period = 7;

            if (derData.size() % 7 != 0)
            {
                if (derData.size() % 6 != 0)
                    break;
                // corrected data
                workOutEqv(derData, enhData);
                ptrData = &enhData;
                if (eqvStrainNotWritten == 0
                    && component == 6
                    && (h_esol_->getIValue() == 3 // elastic strain
                        || h_esol_->getIValue() == 6) // thermal strain
                    )
                {
                    sendWarning("Equivalent strain was not found for some element(s)...");
                    sendWarning("... for elastic or thermal strains a Poisson constant value of 0.5 was used, ...");
                    sendWarning("... the results may accordingly show discrepancies with the ANSYS postprocessor for these elements.");
                    eqvStrainNotWritten = 1;
                }
            }

            switch (topBot)
            {
            case BOTTOM:
                offset = (*ptrData).size() / 2;
            // break;
            case TOP:
                numTimes = (*ptrData).size() / (2 * period);
                for (node = 0; node < numTimes; ++node)
                {
                    ret.push_back((*ptrData)[node * period + component + offset]);
                }
                break;
            case AVERAGE:
                numTimes = (*ptrData).size() / period;
                for (node = 0; node < numTimes; ++node)
                {
                    ret.push_back(.5 * ((*ptrData)[node * period + component] + (*ptrData)[node * period + component + (*ptrData).size() / 2]));
                }
                break;
            }
            break;

        default:
            break;
        }
        break;

    case 6: // Field flux
        component = h_thermalFlux_->getIValue() - 1;
        switch (elem_db_.getStressSupport(routine))
        {
        case ANSYS::THERMAL_SOLID:
        case ANSYS::THERMAL_PLANE: // eeps, das ist wie THERMAL_SOLID !!!!!!!!
            period = 3;
            break;

        default:
            sendWarning("Unhandled value for StressSupport_[routine] in ReadANSYS::ProcessDerivedField");
            ret.push_back(ReadRST::FImpossible_);
            return ret;
            break;
        }
        numTimes = derData.size() / period;
        if (component != 3) // not a vector
        {
            for (node = 0; node < numTimes; ++node)
            {
                ret.push_back(derData[node * period + component]);
            }
        }
        else // vector
        {
            for (int j = 0; j < 3; j++)
            {
                for (node = 0; node < numTimes; ++node)
                {
                    ret.push_back(derData[node * period + j]);
                }
            }
        }
        break;

    case 7: // volume and energy
        component = h_vol_energy_->getIValue();
        switch (component)
        {
        case 0: // volume
            ret.push_back(derData[0]);
            break;

        case 1: // sene
            ret.push_back(derData[1]);
            break;

        case 2: // kene
            ret.push_back(derData[3]);
            break;
        }
        break;

    case 8: // magnetic flux density
        component = h_mag_flux_dens_->getIValue() + 1;
        if (component != 5 && component != 1) // not bsum and no vector
        {
            ret.push_back(derData[component - 2]);
        }
        else if (component != 1) // bsum
        {
            ret.push_back(sqrt(derData[0] * derData[0] + derData[1] * derData[1] + derData[2] * derData[2]));
        }
        else // vector
        {
            ret.push_back(derData[0]);
            ret.push_back(derData[1]);
            ret.push_back(derData[2]);
        }
        break;

    case 9:
        for (node = 0; node < derData.size(); ++node)
        {
            ret.push_back(derData[node]);
        }
        break;

    default:
        ret.push_back(ReadRST::FImpossible_);
        break;
    }
    return ret;
}

void
ReadANSYS::MakeGridAndObjectsElems(const std::string &gridName,
                                   std::vector<int> &e_l,
                                   std::vector<int> &v_l,
                                   std::vector<float> &x_l,
                                   std::vector<float> &y_l,
                                   std::vector<float> &z_l,
                                   std::vector<int> &t_l,
                                   const std::string &dataName,
                                   const float *field0,
                                   const float *field1,
                                   const float *field2,
                                   const std::string &matName,
                                   const int *materials,
                                   const std::string &nodeIndName,
                                   std::vector<coDistributedObject *> &grid_set_list,
                                   std::vector<coDistributedObject *> &data_set_list,
                                   std::vector<coDistributedObject *> &mat_set_list,
                                   std::vector<coDistributedObject *> &node_decode_list,
                                   bool data_per_elem)
{
    // first we have to make a list of marks for all nodes...
    // bit 0 -> on: bit in geometry
    // bit 1 -> on: bit in geometry and has data
    // bit 2 -> on: bit in geometry and is in grid with data
    // bit 3 -> on: bit in geometry and is in grid without data
    int *marks = new int[x_l.size()];
    int *markElems = new int[e_l.size()];
    memset(marks, 0, sizeof(int) * x_l.size());
    memset(markElems, 0, sizeof(int) * e_l.size());
    int elem, vert;
    for (elem = 0; elem < e_l.size(); ++elem)
    {
        for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
        {
            marks[v_l[e_l[elem] + vert]] |= 1; // bit 0
        }
        if (data_per_elem)
        {
            if (field0[v_l[e_l[elem]]] != ReadRST::FImpossible_)
            {
                markElems[elem] = 1;

                for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
                {
                    marks[v_l[e_l[elem] + vert]] |= 6; // bit 1 and 2
                }
            }
            else
            {
                for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
                {
                    marks[v_l[e_l[elem] + vert]] |= 8; // bit 3
                }
            }
        }
        else // data per node
        {
            bool take_it = true;
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                if (field0[v_l[e_l[elem] + vert]] == ReadRST::FImpossible_)
                {
                    take_it = false;
                }
            }
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                if (take_it)
                {
                    marks[v_l[e_l[elem] + vert]] |= 6;
                    if (field0[v_l[e_l[elem] + vert]] == ReadRST::FImpossible_)
                    {
                        marks[v_l[e_l[elem] + vert]] |= 8;
                    }
                    markElems[elem] = 1;
                }
                else
                {
                    marks[v_l[e_l[elem] + vert]] |= 8;
                }
            }
        }
    }

    // make a map for decoding nodes with data and without data
    std::vector<int> nodes_with_data;
    std::vector<int> nodes_without_data;
    std::vector<float> x_l_data;
    std::vector<float> y_l_data;
    std::vector<float> z_l_data;
    std::vector<float> x_l_no_data;
    std::vector<float> y_l_no_data;
    std::vector<float> z_l_no_data;
    std::vector<float> field_data[3];

    int node;
    for (node = 0; node < x_l.size(); ++node)
    {
        if (marks[node] & 4)
        {
            nodes_with_data.push_back(node);
            x_l_data.push_back(x_l[node]);
            y_l_data.push_back(y_l[node]);
            z_l_data.push_back(z_l[node]);
            if (!h_output_node_decode_->getIValue() && !data_per_elem)
            {
                field_data[0].push_back(field0[node]);
                if (field1)
                {
                    field_data[1].push_back(field1[node]);
                    field_data[2].push_back(field2[node]);
                }
            }
        }
        if (marks[node] & 8)
        {
            nodes_without_data.push_back(node);
            x_l_no_data.push_back(x_l[node]);
            y_l_no_data.push_back(y_l[node]);
            z_l_no_data.push_back(z_l[node]);
        }
    }
    Map1D nodesDataDecode(nodes_with_data.size(), &nodes_with_data[0]);
    int *np = (nodes_without_data.size() > 0) ? &nodes_without_data[0] : NULL;
    Map1D nodesNoDataDecode(nodes_without_data.size(), np);

    // now create data for grid with data (and without)
    std::vector<int> e_l_data;
    std::vector<int> v_l_data;
    std::vector<int> t_l_data;
    std::vector<int> m_l_data;
    std::vector<int> e_l_no_data;
    std::vector<int> v_l_no_data;
    std::vector<int> t_l_no_data;
    std::vector<int> m_l_no_data;

    int v_count_data = 0;
    int v_count_no_data = 0;

    for (elem = 0; elem < e_l.size(); ++elem)
    {
        if (markElems[elem])
        {

            if (!h_output_node_decode_->getIValue() && data_per_elem)
            {
                field_data[0].push_back(field0[elem]);
                if (field1)
                {
                    field_data[1].push_back(field1[elem]);
                    field_data[2].push_back(field2[elem]);
                }
            }
            e_l_data.push_back(v_count_data);
            m_l_data.push_back(materials[elem]);
            v_count_data += UnstructuredGrid_Num_Nodes[t_l[elem]];
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                v_l_data.push_back(nodesDataDecode[v_l[e_l[elem] + vert]]);
            }
            t_l_data.push_back(t_l[elem]);
        }
        else
        {
            e_l_no_data.push_back(v_count_no_data);
            m_l_no_data.push_back(materials[elem]);
            v_count_no_data += UnstructuredGrid_Num_Nodes[t_l[elem]];
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                v_l_no_data.push_back(nodesNoDataDecode[v_l[e_l[elem] + vert]]);
            }
            t_l_no_data.push_back(t_l[elem]);
        }
    }

    // now create objects
    coCellToVert coInterp;

    if (t_l_no_data.size() == 0) // the whole grid has data
    {
        coDoUnstructuredGrid *p_grid = new coDoUnstructuredGrid(gridName,
                                                                e_l_data.size(),
                                                                v_l_data.size(), x_l_data.size(),
                                                                &e_l_data[0],
                                                                &v_l_data[0],
                                                                &x_l_data[0],
                                                                &y_l_data[0],
                                                                &z_l_data[0],
                                                                &t_l_data[0]);
        p_grid->addAttribute("COLOR", "White");
        grid_set_list.push_back(p_grid);
        // Trivial node decode indices, if required
        if (h_output_node_decode_->getIValue())
        {
            int dimArray = x_l_data.size();
            coDistributedObject *intArray = new coDoIntArr(nodeIndName, 1, &dimArray);
            node_decode_list.push_back(intArray);
            int *trivial_codes = dynamic_cast<coDoIntArr *>(intArray)->getAddress();
            int point;
            for (point = 0; point < x_l_data.size(); ++point)
            {
                trivial_codes[point] = point;
            }
        }
        else
        {
            if (!field1)
            {
                int sz = field_data[0].size();
                if (sz != 0 && data_per_elem && p_vertex_based_->getValue())
                {
                    data_set_list.push_back(coInterp.interpolate(p_grid, 1,
                                                                 sz, &field_data[0][0], NULL, NULL, dataName.c_str()));
                }
                else
                {
                    data_set_list.push_back(new coDoFloat(dataName,
                                                          field_data[0].size(),
                                                          &field_data[0][0]));
                }
            }
            else
            {
                int sz = field_data[0].size();
                if (sz != 0 && data_per_elem && p_vertex_based_->getValue())
                {
                    data_set_list.push_back(coInterp.interpolate(p_grid, 3,
                                                                 sz,
                                                                 &field_data[0][0],
                                                                 &field_data[1][0],
                                                                 &field_data[2][0],
                                                                 dataName.c_str()));
                }
                else
                {
                    data_set_list.push_back(new coDoVec3(dataName,
                                                         field_data[0].size(),
                                                         &field_data[0][0],
                                                         &field_data[1][0],
                                                         &field_data[2][0]));
                }
            }
            if (!p_file_name_->isConnected())
            {
                data_set_list[data_set_list.size() - 1]->addAttribute("SPECIES", p_esol_->getActLabel());
            }
        }
        int matSize = m_l_data.size();
        coDoIntArr *matOut = new coDoIntArr(matName, 1, &matSize);
        memcpy(matOut->getAddress(), &m_l_data[0], m_l_data.size() * sizeof(int));
        mat_set_list.push_back(matOut);
    }
    else // there is a grid without data
    {
        std::string gridNameData(gridName);
        gridNameData += "_data";
        std::string gridNameNoData(gridName);
        gridNameNoData += "_no_data";
        std::string dataNameData(dataName);
        dataNameData += "_data";
        std::string dataNameNoData(dataName); // for the dummy
        dataNameNoData += "_no_data";
        std::string matNameData(matName);
        matNameData += "_data";
        std::string matNameNoData(matName);
        matNameNoData += "_no_data";

        coDistributedObject *grid_list[3];
        coDistributedObject *data_list[3];
        coDistributedObject *mat_list[3];

        grid_list[2] = NULL;
        data_list[2] = NULL;
        mat_list[2] = NULL;

        // grid
        grid_list[0] = new coDoUnstructuredGrid(gridNameData, e_l_data.size(),
                                                v_l_data.size(), x_l_data.size(),
                                                &e_l_data[0],
                                                &v_l_data[0],
                                                &x_l_data[0],
                                                &y_l_data[0],
                                                &z_l_data[0],
                                                &t_l_data[0]);
        grid_list[1] = new coDoUnstructuredGrid(gridNameNoData, e_l_no_data.size(),
                                                v_l_no_data.size(), x_l_no_data.size(),
                                                &e_l_no_data[0],
                                                &v_l_no_data[0],
                                                &x_l_no_data[0],
                                                &y_l_no_data[0],
                                                &z_l_no_data[0],
                                                &t_l_no_data[0]);
        grid_list[0]->addAttribute("COLOR", "White");
        grid_list[1]->addAttribute("COLOR", "White");
        // material
        int matSize = m_l_data.size();
        coDoIntArr *matOutData = new coDoIntArr(matNameData, 1, &matSize);
        memcpy(matOutData->getAddress(), &m_l_data[0], m_l_data.size() * sizeof(int));
        mat_list[0] = matOutData;
        matSize = m_l_no_data.size();
        coDoIntArr *matOutNoData = new coDoIntArr(matNameNoData, 1, &matSize);
        memcpy(matOutNoData->getAddress(), &m_l_no_data[0], m_l_no_data.size() * sizeof(int));
        mat_list[1] = matOutNoData;

        grid_set_list.push_back(new coDoSet(gridName, grid_list));
        mat_set_list.push_back(new coDoSet(matName, mat_list));
        // data
        if (!h_output_node_decode_->getIValue())
        {
            if (!field1)
            {
                int sz = field_data[0].size();
                if (sz != 0 && data_per_elem && p_vertex_based_->getValue())
                {
                    data_list[0] = coInterp.interpolate(grid_list[0], 1,
                                                        sz,
                                                        &field_data[0][0], NULL, NULL,
                                                        dataNameData.c_str());
                }
                else
                {
                    data_list[0] = new coDoFloat(dataNameData,
                                                 field_data[0].size(),
                                                 &field_data[0][0]);
                }
                data_list[1] = new coDoFloat(dataNameNoData, 0);
            }
            else
            {
                int sz = field_data[0].size();
                if (sz != 0 && data_per_elem && p_vertex_based_->getValue())
                {
                    data_list[0] = coInterp.interpolate(grid_list[0], 3,
                                                        sz,
                                                        &field_data[0][0],
                                                        &field_data[1][0],
                                                        &field_data[2][0],
                                                        dataNameData.c_str());
                }
                else
                {
                    data_list[0] = new coDoVec3(dataNameData,
                                                field_data[0].size(),
                                                &field_data[0][0],
                                                &field_data[1][0],
                                                &field_data[2][0]);
                }
                data_list[1] = new coDoVec3(dataNameNoData, 0);
            }
            if (!p_file_name_->isConnected())
            {
                if (data_list[0])
                    data_list[0]->addAttribute("SPECIES", p_esol_->getActLabel());
                if (data_list[1])
                    data_list[1]->addAttribute("SPECIES", p_esol_->getActLabel());
            }
            data_set_list.push_back(new coDoSet(dataName, data_list));
        }
        // node decode indices if required
        else
        {
            std::string nodeIndNameData(nodeIndName);
            nodeIndNameData += "_data";
            std::string nodeIndNameNoData(nodeIndName);
            nodeIndNameNoData += "_no_data";
            coDistributedObject *node_ind_list[3];
            node_ind_list[2] = NULL;
            int numData = nodes_with_data.size();
            node_ind_list[0] = new coDoIntArr(nodeIndNameData, 1, &numData, &nodes_with_data[0]);
            int numNoData = nodes_without_data.size();
            node_ind_list[1] = new coDoIntArr(nodeIndNameNoData, 1, &numNoData, &nodes_without_data[0]);
            node_decode_list.push_back(new coDoSet(nodeIndName, node_ind_list));
        }
    }
    delete[] marks;
    delete[] markElems;
}

double
EqvInvariant(const double *symTens)
{
    double trace3 = (symTens[0] + symTens[1] + symTens[2]) / 3.0;
    double ret = (symTens[0] - trace3) * (symTens[0] - trace3);
    ret += (symTens[1] - trace3) * (symTens[1] - trace3);
    ret += (symTens[2] - trace3) * (symTens[2] - trace3);
    ret += 2.0 * symTens[3] * symTens[3];
    ret += 2.0 * symTens[4] * symTens[4];
    ret += 2.0 * symTens[5] * symTens[5];
    return ret;
}

void
ReadANSYS::workOutEqv(std::vector<double> &derData, std::vector<double> &enhData)
{
    enhData.clear();
    int punkt;
    int n_punkt = derData.size() / 6;
    for (punkt = 0; punkt < n_punkt; ++punkt)
    {
        int comp;
        for (comp = 0; comp < 6; ++comp)
        {
            enhData.push_back(derData[6 * punkt + comp]);
        }
        double theInvariant = EqvInvariant(&derData[0] + 6 * punkt);
        // Poisson constant
        enhData.push_back(sqrt(1.5 * theInvariant) / (1 + 0.5));
        // assumed to be 0.5
    }
}
