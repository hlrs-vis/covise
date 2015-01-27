/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/coviseCompat.h>
#include <do/coDoAbstractStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <algorithm>
#include "Transform.h"
#include <vector>
#include <string>
#include <functional>

Transform::Transform(int argc, char *argv[])
    : coSimpleModule(argc, argv, "transform geometry")
    , transformationAsAttribute(false)
{

    p_type_ = paraSwitch("Transform", "Please enter your choice");
    paraCase("Mirror");
    p_mirror_normal_ = addFloatVectorParam("normal_of_mirror_plane",
                                           "normal of mirror-plane");
    p_mirror_normal_->setValue(0.0, 0.0, 1.0);
    p_mirror_dist_ = addFloatParam("distance_to_origin",
                                   "distance to the origin");
    p_mirror_and_original_ = addBooleanParam("MirroredAndOriginal",
                                             "Transformed object(s) and original");
    p_mirror_and_original_->setValue(0);
    paraEndCase();
    paraCase("Translate");
    p_trans_vertex_ = addFloatVectorParam("vector_of_translation",
                                          "transformation");
    p_trans_vertex_->setValue(0.0, 0.0, 0.0);
    paraEndCase();
    paraCase("Rotate");
    p_rotate_normal_ = addFloatVectorParam("axis_of_rotation",
                                           "axis of rotation");
    p_rotate_normal_->setValue(0.0, 0.0, 1.0);
    p_rotate_vertex_ = addFloatVectorParam("one_point_on_the_axis",
                                           "transformation");
    p_rotate_vertex_->setValue(0.0, 0.0, 0.0);
    p_rotate_scalar_ = addFloatParam("angle_of_rotation",
                                     "angle of rotation");
    p_rotate_scalar_->setValue(1.0);
    paraEndCase();
    paraCase("Scale");
    const char *ScaleType[] = { "Uniform", "X-axis", "Y-axis", "Z-axis" };
    p_scale_type_ = addChoiceParam("scale_type", "Scale type");
    p_scale_type_->setValue(4, ScaleType, 0);
    p_scale_scalar_ = addFloatParam("scaling_factor", "scaling factor");
    p_scale_scalar_->setValue(1.0);
    p_scale_vertex_ = addFloatVectorParam("new_origin", "new origin");
    p_scale_vertex_->setValue(0.0, 0.0, 0.0);
    paraEndCase();
    paraCase("MultiRot");
    p_multirot_normal_ = addFloatVectorParam("axis_of_multi_rotation",
                                             "axis of rotation");
    p_multirot_normal_->setValue(0.0, 0.0, 1.0);
    p_multirot_vertex_ = addFloatVectorParam("_one_point_on_the_axis",
                                             "transformation");
    p_multirot_vertex_->setValue(0.0, 0.0, 0.0);
    p_multirot_scalar_ = addFloatParam("angle_of_multi_rotation",
                                       "angle of rotation");
    p_multirot_scalar_->setValue(1.0);
    p_number_ = addInt32Param("number_of_rotations", "number of rotations");
    p_number_->setValue(1);
    paraEndCase();
    paraCase("Tile");
    const char *TilePlane[] = { "XY", "YZ", "ZX" };
    p_tiling_plane_ = addChoiceParam("TilingPlane", "Tiling plane");
    p_tiling_plane_->setValue(3, TilePlane, 0);

    p_flipTile_ = addBooleanParam("flipTile",
                                  "Flip or else translate tile");
    p_flipTile_->setValue(1);

    p_tiling_min_ = addInt32VectorParam("TilingMin", "Tiling pattern");
    const long defaultMinTilingPattern[] = { 0, 0 };
    p_tiling_min_->setValue(2, defaultMinTilingPattern);

    p_tiling_max_ = addInt32VectorParam("TilingMax", "Tiling pattern");
    const long defaultMaxTilingPattern[] = { 3, 3 };
    p_tiling_max_->setValue(2, defaultMaxTilingPattern);

    paraEndCase();
    paraCase("TimeDependent");
    p_time_matrix_ = addFileBrowserParam("EUC_file", "Euclidian motion");
    p_time_matrix_->setValue("/tmp/foo.euc", "*.euc");
    paraEndCase();
    paraEndSwitch();

    int data_port;
    for (data_port = 0; data_port < NUM_DATA_IN_PORTS; ++data_port)
    {
        const char *TileChoices[] = {
            "TrueVectorOrScalar",
            "PseudoVectorOrScalar",
            "Displacements"
        };
        std::string dataPortName("InDataType");
        std::string dataPortDesc("Input data type");
        char buf[64];
        char buf_desc[64];
        sprintf(buf, "_%d", data_port);
        sprintf(buf_desc, " %d", data_port);
        dataPortName += buf;
        dataPortDesc += buf;
        p_dataType_[data_port] = addChoiceParam(dataPortName.c_str(), dataPortDesc.c_str());
        p_dataType_[data_port]->setValue(3, TileChoices, 0);
    }

    p_create_set_ = addBooleanParam("createSet", "create sets for multiple transformations");
    p_create_set_->setValue(1);

    p_geo_in_ = addInputPort("geo_in", "Polygons|TriangleStrips|Points|Lines|UnstructuredGrid|UniformGrid|RectilinearGrid|StructuredGrid", "polygon/grid input");
    p_geo_out_ = addOutputPort("geo_out", "Polygons|TriangleStrips|Points|Lines|UnstructuredGrid|UniformGrid|RectilinearGrid|StructuredGrid", "polygon/grid output");

    for (data_port = 0; data_port < NUM_DATA_IN_PORTS; data_port++)
    {
        char portname[64];
        sprintf(portname, "data_out%d", data_port);
        p_data_out_[data_port] = addOutputPort(portname, "Float|Vec3", "data output");
        sprintf(portname, "data_in%d", data_port);
        p_data_in_[data_port] = addInputPort(portname, "Float|Vec3", "data output");
        p_data_in_[data_port]->setRequired(0);
        p_data_out_[data_port]->setDependencyPort(p_data_in_[data_port]);
    }
    // This port seems quite useless...
    // p_angle_out_ = addOutputPort("angle_out","Float",
    //                          "rotate-angle for other modules");

    // we have to care about attributes because of these reasons
    // 1. Some options give rise to sets (multirotate, mirror, tile)
    // 2. There are especial attributes: ROTATE_POINT, etc
    // 3. Sometimes input objects are reused
    setCopyNonSetAttributes(0);
    preHandleFailed_ = 1;
    lagrange_ = 0;
}

// the hparams_ objects look almost like parameters, but may
// use the values gotten from a TRANSFORM attribute
void
Transform::postInst()
{
    hparams_.push_back(h_mirror_dist_ = new coHideParam(p_mirror_dist_));
    hparams_.push_back(h_multirot_scalar_ = new coHideParam(p_multirot_scalar_));
    hparams_.push_back(h_rotate_scalar_ = new coHideParam(p_rotate_scalar_));
    hparams_.push_back(h_scale_scalar_ = new coHideParam(p_scale_scalar_));
    hparams_.push_back(h_mirror_and_original_ = new coHideParam(p_mirror_and_original_));
    hparams_.push_back(h_trans_vertex_ = new coHideParam(p_trans_vertex_));
    hparams_.push_back(h_scale_vertex_ = new coHideParam(p_scale_vertex_));
    hparams_.push_back(h_rotate_vertex_ = new coHideParam(p_rotate_vertex_));
    hparams_.push_back(h_multirot_vertex_ = new coHideParam(p_multirot_vertex_));
    hparams_.push_back(h_mirror_normal_ = new coHideParam(p_mirror_normal_));
    hparams_.push_back(h_rotate_normal_ = new coHideParam(p_rotate_normal_));
    hparams_.push_back(h_multirot_normal_ = new coHideParam(p_multirot_normal_));
    hparams_.push_back(h_number_ = new coHideParam(p_number_));
    hparams_.push_back(h_type_ = new coHideParam(p_type_));
    hparams_.push_back(h_tiling_plane_ = new coHideParam(p_tiling_plane_));
    hparams_.push_back(h_tiling_min_ = new coHideParam(p_tiling_min_));
    hparams_.push_back(h_tiling_max_ = new coHideParam(p_tiling_max_));
    hparams_.push_back(h_flipTile_ = new coHideParam(p_flipTile_));
    int port;
    for (port = 0; port < NUM_DATA_IN_PORTS; ++port)
    {
        hparams_.push_back(h_dataType_[port] = new coHideParam(p_dataType_[port]));
    }
}

void
Transform::useTransformAttribute(const coDistributedObject *inGeo)
{
    const char *wert;
    if (inGeo == NULL)
    {
        return;
    }
    wert = inGeo->getAttribute("TRANSFORM");
    if (wert == NULL) // perhaps the attribute is hidden in a set structure
    {
        if (inGeo->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *setList = ((const coDoSet *)(inGeo))->getAllElements(&no_elems);
            int elem;
            for (elem = 0; elem < no_elems; ++elem)
            {
                useTransformAttribute(setList[elem]);
            }
        }
        return;
    }
    std::istringstream pvalues(wert);
    char *value = new char[strlen(wert) + 1];
    while (pvalues.getline(value, strlen(wert) + 1))
    {
        int param;
        for (param = 0; param < hparams_.size(); ++param)
        {
            hparams_[param]->load(value);
        }
    }
    delete[] value;
}

struct IsSpace : public std::unary_function<char, bool>
{
    bool operator()(const char &c) const
    {
        return (isspace(c) != 0);
    }
};

static void
EliminateTrailingSpaces(char *buf)
{
    if (buf == NULL)
        return;
    string loc_buf(buf);
    string::reverse_iterator ri(std::find_if(loc_buf.rbegin(), loc_buf.rend(),
                                             std::not1(IsSpace())));
    string::iterator it(ri.base());
    string correct_buf(loc_buf.begin(), it);
    strcpy(buf, correct_buf.c_str());
}

// preHandleObjects is only required in hte tiling case,
// then we have to work out the bounding box and occasionally
// also the displacements at the corners
void
Transform::preHandleObjects(coInputPort **inPorts)
{
    coInputPort *geometry = inPorts[0];

    int param;
    for (param = 0; param < hparams_.size(); ++param)
    {
        hparams_[param]->reset();
    }
    useTransformAttribute(geometry->getCurrentObject());

    preHandleFailed_ = 1;
    lookUp_ = 0;
    BBoxes_.clean(h_tiling_plane_->getIValue());

    // which data port has been declared to be 'Displacements'?
    // normaly a number >= 0, it is -1 if no displacements are
    // available, -2 if several data ports have been tagged
    // to have displacements
    int displacementsPort = lookForDisplacements();

    if (displacementsPort != -1 && h_type_->getIValue() != TYPE_TILE)
    {
        sendWarning("The option 'Displacements' is only permitted when tiling");
        return; // STOP_PIPELINE;
    }
    else if (TYPE_TILE != h_type_->getIValue())
    {
        if (TIME_DEPENDENT == h_type_->getIValue())
        {
            dynamicRefSystem_.clear();
            // open file referred to by p_time_matrix_
            ifstream infile(p_time_matrix_->getValue());
            if (!infile)
            {
                sendWarning("Could not open time-dependent matrix file");
                return;
            }
            const int size = 1024;
            char buf[size];
            while (infile.getline(buf, size, '\n'))
            {
                EliminateTrailingSpaces(buf);
                if (strcmp(buf, "*MATRIX") == 0)
                {
                    Matrix theMatrix;
                    int zeile = 0;
                    while (infile.getline(buf, size, '\n'))
                    {
                        float v[3];
                        istringstream line(buf);
                        if (line >> v[0] >> v[1] >> v[2])
                        {
                            theMatrix.setMatrix(v, zeile++);
                        }
                        if (zeile == 4)
                        {
                            theMatrix.setFlags();
                            dynamicRefSystem_.push_back(theMatrix);
                            break;
                        }
                    }
                    if (zeile != 4)
                    {
                        sendWarning("Syntax error in time-dependent matrix file");
                        return;
                    }
                } // if MATRIX
            } // while getline
        } // if TIME_DEPENDENT
        // if no tiling we do not need preHandleObjects at all: this is OK
        preHandleFailed_ = 0;
        return; // CONTINUE_PIPELINE;
    }
    // From now on we know we are tiling
    if (displacementsPort == -2)
    {
        sendWarning("You may not tile using displacements twice");
        return; // STOP_PIPELINE;
    }

    coInputPort *displacements = (displacementsPort >= 0) ? inPorts[displacementsPort + 1] : NULL;

    // only 2 objects (at most) are required: geometry and displacements
    const coDistributedObject *inGeometry = geometry->getCurrentObject();
    const coDistributedObject *inDisplacements = (displacements != NULL) ? displacements->getCurrentObject() : NULL;

    if (inGeometry == NULL
        || !inGeometry->objectOk())
    {
        sendWarning("Input geometry not available or not OK");
        return; // STOP_PIPELINE;
    }
    else if (inGeometry->isType("SETELE"))
    {
        coDoSet *in_set_geom = (coDoSet *)(inGeometry);
        coDoSet *in_set_disp = (coDoSet *)(inDisplacements);
        int no_set_elems;
        // dynamic set
        if (in_set_geom->getAttribute("TIMESTEP"))
        {
            const coDistributedObject *const *geom_objs = NULL;
            const coDistributedObject *const *disp_objs = NULL;
            geom_objs = in_set_geom->getAllElements(&no_set_elems);
            if (in_set_disp)
            {
                int no_set_elems_disp;
                disp_objs = in_set_disp->getAllElements(&no_set_elems_disp);
                if (no_set_elems_disp != no_set_elems)
                {
                    sendError("Input geometry set structure does \
                           not match displacement set structure");
                    return; // STOP_PIPELINE;
                }
            }
            //make room for all time steps
            BBoxes_.prepare(no_set_elems);
            int time;
            for (time = 0; time < no_set_elems; ++time)
            {
                const coDistributedObject *dataObj = (disp_objs != NULL) ? disp_objs[time] : NULL;
                preHandleFailed_ = BBoxes_.FillBBox(geom_objs[time],
                                                    dataObj, time);
                if (preHandleFailed_)
                {
                    break;
                }
            }
        }
        // static set
        else
        {
            BBoxes_.prepare(1);
            preHandleFailed_ = BBoxes_.FillBBox(inGeometry, inDisplacements, 0);
        }
    }
    // a static non-set object
    else
    {
        BBoxes_.prepare(1);
        preHandleFailed_ = BBoxes_.FillBBox(inGeometry, inDisplacements, 0);
    }
    if (preHandleFailed_)
    {
        sendWarning("Could not determine bounding box, or problem when reading displacements, if any.");
    }
    return; // CONTINUE_PIPELINE;
}

// we may always access variable lookUp_ to know
// what the time is
void
Transform::setIterator(coInputPort **inPorts, int t)
{
    const char *dataType;

    dataType = (inPorts[0]->getCurrentObject())->getType();
    if (strcmp(dataType, "SETELE") == 0
        && inPorts[0]->getCurrentObject()->getAttribute("TIMESTEP"))
    {
        lookUp_ = t;
    }
    return; // CONTINUE_PIPELINE;
}

void
Transform::postHandleObjects(coOutputPort **)
{
    BBoxes_.clean(h_tiling_plane_->getIValue());
    return; // CONTINUE_PIPELINE;
}

// Create all involved transformations
Matrix *
Transform::fillTransformations(int *numTransformations)
{
    // determine number of transformations
    switch (h_type_->getIValue())
    {
    case TYPE_MIRROR:
    case TYPE_TRANSLATE:
    case TYPE_ROTATE:
    case TYPE_SCALE:
        *numTransformations = 1;
        break;
    case TYPE_MULTI_ROTATE:
        *numTransformations = h_number_->getIValue();
        if (*numTransformations <= 0)
        {
            sendError("The number of rotations is not positive");
        }
        break;
    case TYPE_TILE:
        *numTransformations = (h_tiling_max_->getIValue(1) - h_tiling_min_->getIValue(1) + 1) * (h_tiling_max_->getIValue(0) - h_tiling_min_->getIValue(0) + 1);
        if (h_tiling_max_->getIValue(1) - h_tiling_min_->getIValue(1) < 0 || h_tiling_max_->getIValue(0) - h_tiling_min_->getIValue(0) < 0)
        {
            sendError("Tiling parameters are not correct, please make sure that maximum values are greater than the minimum ones");
        }
        break;
    case TIME_DEPENDENT:
        *numTransformations = 1;
        break;
    default:
        sendError("Please, choose transformation type");
        *numTransformations = 0;
        return NULL;
        break;
    }
    bool keep_original = h_mirror_and_original_->getIValue();
    sendInfo("keeping original: %i", (int)keep_original);
    if (keep_original && h_type_->getIValue() == TYPE_TILE)
    {
        if (h_tiling_min_->getIValue(0) <= 0 && h_tiling_max_->getIValue(0) >= 0
            && h_tiling_min_->getIValue(1) <= 0 && h_tiling_max_->getIValue(1) >= 0)
        {
            // original already included in tiles
            keep_original = false;
        }
    }
    if (keep_original)
    {
        *numTransformations += 1;
    }
    if (*numTransformations <= 0)
    {
        return NULL;
    }
    sendInfo("keeping original: %i, num trans=%i", (int)keep_original, *numTransformations);
    Matrix *retMatrices = new Matrix[*numTransformations]; // initialized to identity matrices
    Matrix *retMatrix = keep_original ? &retMatrices[1] : &retMatrices[0];
    // now fill the matrices
    // this ought to be as clear and transparent as
    // the soup in an orphanage
    switch (h_type_->getIValue())
    {
    case TYPE_MIRROR:
    {
        float normal[3];
        normal[0] = h_mirror_normal_->getFValue(0);
        normal[1] = h_mirror_normal_->getFValue(1);
        normal[2] = h_mirror_normal_->getFValue(2);
        retMatrix->MirrorMatrix(h_mirror_dist_->getFValue(), normal);
    }
    break;
    case TYPE_TRANSLATE:
        retMatrix->TranslateMatrix(h_trans_vertex_->getFValue(0),
                                   h_trans_vertex_->getFValue(1),
                                   h_trans_vertex_->getFValue(2));
        break;
    case TYPE_ROTATE:
    {
        float angleDEG = h_rotate_scalar_->getFValue();
        float vertex[3];
        float normal[3];
        h_rotate_vertex_->getFValue(vertex[0], vertex[1], vertex[2]);
        h_rotate_normal_->getFValue(normal[0], normal[1], normal[2]);
        retMatrix->RotateMatrix(angleDEG, vertex, normal);
    }
    break;
    case TYPE_SCALE:
        retMatrix->ScaleMatrix(h_scale_scalar_->getFValue(),
                               h_scale_vertex_->getFValue(0),
                               h_scale_vertex_->getFValue(1),
                               h_scale_vertex_->getFValue(2),
                               p_scale_type_->getValue() + 1);
        break;
    case TYPE_MULTI_ROTATE:
        MultiRotateMatrix(retMatrix);
        break;
    case TYPE_TILE:
        TileMatrix(retMatrix);
        break;
    case TIME_DEPENDENT:
        if (!dynamicRefSystem_.empty())
        {
            *retMatrix = dynamicRefSystem_[lookUp_ % dynamicRefSystem_.size()];
        }
        break;
    }
    return retMatrices;
}

// this ought to be as clear and transparent as
// the soup in an orphanage
void
Transform::MultiRotateMatrix(Matrix *matrix)
{
    int rotat;
    float vertex[3], normal[3];
    h_multirot_vertex_->getFValue(vertex[0], vertex[1], vertex[2]);
    h_multirot_normal_->getFValue(normal[0], normal[1], normal[2]);
    for (rotat = 1; rotat <= h_number_->getIValue(); ++rotat)
    {
        float angleDEG = rotat * h_multirot_scalar_->getFValue();
        matrix[rotat - 1].RotateMatrix(angleDEG, vertex, normal);
    }
}

// One tile is either trivial (object may be reused)
// or is obtained, in the general case, combining
// 2 transformations, which may be either translations or mirroring.
void
Transform::TileMatrix(Matrix *retMatrix)
{
    // check that tesselation limits are sensible
    if (h_tiling_min_->getIValue(0) > h_tiling_max_->getIValue(0)
        || h_tiling_min_->getIValue(1) > h_tiling_max_->getIValue(1))
    {
        sendError("Check tiling limits");
        return;
    }
    int tile_x, tile_y; // here x, y have the meaning of local coordinates!!!
    int countMatrix = 0;
    const float *bbox = BBoxes_.getBBox(lookUp_);
    for (tile_x = h_tiling_min_->getIValue(0); tile_x <= h_tiling_max_->getIValue(0);
         tile_x++)
    {
        Matrix XMove;
        // translate or mirror in the x-local direction
        // mirror
        if ((tile_x % 2 != 0) && h_flipTile_->getIValue())
        {
            float mirror_normal[3] = { 0.0, 0.0, 0.0 };
            float distance;
            switch (h_tiling_plane_->getIValue())
            {
            case 1: // XY = local xy
                mirror_normal[0] = 1.0;
                distance = 0.5f * tile_x * (bbox[BBoxes::MAX_X] - bbox[BBoxes::MIN_X]) + 0.5f * (bbox[BBoxes::MAX_X] + bbox[BBoxes::MIN_X]);
                if (lagrange_)
                {
                    distance -= 0.5f * tile_x * (bbox[BBoxes::UXB] - bbox[BBoxes::UXA]) + 0.5f * (bbox[BBoxes::UXB] + bbox[BBoxes::UXA]);
                }
                break;
            case 2: // YZ = locl xy
                mirror_normal[1] = 1.0;
                distance = 0.5f * tile_x * (bbox[BBoxes::MAX_Y] - bbox[BBoxes::MIN_Y]) + 0.5f * (bbox[BBoxes::MAX_Y] + bbox[BBoxes::MIN_Y]);
                if (lagrange_)
                {
                    distance -= 0.5f * tile_x * (bbox[BBoxes::UYB] - bbox[BBoxes::UYA]) + 0.5f * (bbox[BBoxes::UYB] + bbox[BBoxes::UYA]);
                }
                break;
            case 3: // ZX = local xy
                mirror_normal[2] = 1.0;
                distance = 0.5f * tile_x * (bbox[BBoxes::MAX_Z] - bbox[BBoxes::MIN_Z]) + 0.5f * (bbox[BBoxes::MAX_Z] + bbox[BBoxes::MIN_Z]);
                if (lagrange_)
                {
                    distance -= 0.5f * tile_x * (bbox[BBoxes::UZB] - bbox[BBoxes::UZA]) + 0.5f * (bbox[BBoxes::UYB] + bbox[BBoxes::UZA]);
                }
                break;
            default:
                return;
                break;
            }
            XMove.MirrorMatrix(distance, mirror_normal);
        }
        else // translate
        {
            if (h_tiling_plane_->getIValue() == 0) // XY idem
            {
                XMove.TranslateMatrix(
                    tile_x * (bbox[BBoxes::MAX_X] - bbox[BBoxes::MIN_X]) - lagrange_ * tile_x * (bbox[BBoxes::UXB] - bbox[BBoxes::UXA]),
                    0, 0);
            }
            // YZ idem
            else if (h_tiling_plane_->getIValue() == 1)
            {
                XMove.TranslateMatrix(0,
                                      tile_x * (bbox[BBoxes::MAX_Y] - bbox[BBoxes::MIN_Y]) - lagrange_ * tile_x * (bbox[BBoxes::UYB] - bbox[BBoxes::UYA]),
                                      0);
            }
            // ZX idem
            else if (h_tiling_plane_->getIValue() == 2)
            {
                XMove.TranslateMatrix(0, 0,
                                      tile_x * (bbox[BBoxes::MAX_Z] - bbox[BBoxes::MIN_Z]) - lagrange_ * tile_x * (bbox[BBoxes::UZB] - bbox[BBoxes::UZA]));
            }
        }
        // Now follows the second rtansformations, also
        // mirroring or translation
        for (tile_y = h_tiling_min_->getIValue(1); tile_y <= h_tiling_max_->getIValue(1);
             ++tile_y, ++countMatrix)
        {
            if (tile_x == 0 && tile_y == 0)
            {
                continue;
            }
            Matrix &YMove = retMatrix[countMatrix];
            // translate or mirror?
            // mirror
            if ((tile_y % 2 != 0) && h_flipTile_->getIValue())
            {
                float mirror_normal[3] = { 0.0, 0.0, 0.0 };
                float distance;
                switch (h_tiling_plane_->getIValue())
                {
                case 0: // XY
                    distance = 0.5f * tile_y * (bbox[BBoxes::MAX_Y] - bbox[BBoxes::MIN_Y]) + 0.5f * (bbox[BBoxes::MAX_Y] + bbox[BBoxes::MIN_Y]);
                    if (lagrange_)
                    {
                        distance -= 0.5f * tile_y * (bbox[BBoxes::UYB] - bbox[BBoxes::UYA]) + 0.5f * (bbox[BBoxes::UYB] + bbox[BBoxes::UYA]);
                    }
                    mirror_normal[1] = 1.0f;
                    break;
                case 1: // YZ
                    distance = 0.5f * tile_y * (bbox[BBoxes::MAX_Z] - bbox[BBoxes::MIN_Z]) + 0.5f * (bbox[BBoxes::MAX_Z] + bbox[BBoxes::MIN_Z]);
                    if (lagrange_)
                    {
                        distance -= 0.5f * tile_y * (bbox[BBoxes::UZB] - bbox[BBoxes::UZA]) + 0.5f * (bbox[BBoxes::UZB] + bbox[BBoxes::UZA]);
                    }
                    mirror_normal[2] = 1.0f;
                    break;
                case 2: // ZX
                    distance = 0.5f * tile_y * (bbox[BBoxes::MAX_X] - bbox[BBoxes::MIN_X]) + 0.5f * (bbox[BBoxes::MAX_X] + bbox[BBoxes::MIN_X]);
                    if (lagrange_)
                    {
                        distance -= 0.5f * tile_y * (bbox[BBoxes::UXB] - bbox[BBoxes::UXA]) + 0.5f * (bbox[BBoxes::UXB] + bbox[BBoxes::UXA]);
                    }
                    mirror_normal[0] = 1.0f;
                    break;
                default:
                    return;
                    break;
                }
                YMove.MirrorMatrix(distance, mirror_normal);
            }
            else // translate
            {
                if (h_tiling_plane_->getIValue() == 0) // XY
                {
                    YMove.TranslateMatrix(0,
                                          tile_y * (bbox[BBoxes::MAX_Y] - bbox[BBoxes::MIN_Y]) - lagrange_ * tile_y * (bbox[BBoxes::UYB] - bbox[BBoxes::UYA]),
                                          0);
                }
                // YZ
                else if (h_tiling_plane_->getIValue() == 1)
                {
                    YMove.TranslateMatrix(0, 0,
                                          tile_y * (bbox[BBoxes::MAX_Z] - bbox[BBoxes::MIN_Z]) - lagrange_ * tile_y * (bbox[BBoxes::UZB] - bbox[BBoxes::UZA]));
                }
                // ZX
                else if (h_tiling_plane_->getIValue() == 2)
                {
                    YMove.TranslateMatrix(
                        tile_y * (bbox[BBoxes::MAX_X] - bbox[BBoxes::MIN_X]) - lagrange_ * tile_y * (bbox[BBoxes::UXB] - bbox[BBoxes::UXA]),
                        0, 0);
                }
            }
            retMatrix[countMatrix] *= XMove;
        } // for tile_y
    } // for tile_x
}

int
Transform::compute(const char *)
{
    // for all transformations we first
    // construct a list of required transformations
    if (preHandleFailed_)
    {
        sendError("Cannot proceed with the computation");
        return FAIL;
    }
    int numTransformations;
    Matrix *transformations = fillTransformations(&numTransformations);
    Matrix *lagrangeStateTransformations = NULL;
    int displacementsPort = lookForDisplacements();
    if (displacementsPort == -2)
    {
        return FAIL;
    }
    if (displacementsPort >= 0 && h_type_->getIValue() == TYPE_TILE)
    {
        // when tiling and if displacements are available, we need
        // the transformation for the configuration prior to the
        // displacement. The module assumes that the geometry is
        // already displaced. Variable lagrange_ is used to work
        // with the lagrangean description of the model.
        lagrange_ = 1;
        lagrangeStateTransformations = fillTransformations(&numTransformations);
        lagrange_ = 0;
    }

    // output geometry
    std::string outGeoName(p_geo_out_->getObjName());

    std::vector<Geometry> geometry(numTransformations); // empty geometry

    if (numTransformations == 1)
    {
        const coDistributedObject *out_geo = OutputGeometry(outGeoName.c_str(),
                                                            geometry[0],
                                                            p_geo_in_->getCurrentObject(), transformations[0], 0);
        if (out_geo == NULL)
        {
            sendError("Could not create output geometry");
            return FAIL;
        }
        /* FIXME: RedressOrientation might change input objects */
        p_geo_out_->setCurrentObject(const_cast<coDistributedObject *>(out_geo));
    }
    else if (numTransformations > 1)
    {
        // the same as with a single transformation but bunching
        // results in a set
        const coDistributedObject **setGeoList = new const coDistributedObject *[numTransformations + 1];
        setGeoList[numTransformations] = NULL;
        int trans;
        for (trans = 0; trans < numTransformations; ++trans)
        {
            char bufname[64];
            sprintf(bufname, "%s_%d", outGeoName.c_str(), trans);
            setGeoList[trans] = OutputGeometry(bufname, geometry[trans],
                                               p_geo_in_->getCurrentObject(),
                                               transformations[trans], 1);
            if (setGeoList[trans] == NULL)
            {
                sendError("Could not create output geometry");
                int transClean;
                for (transClean = 0; transClean < trans; ++transClean)
                {
                    // do not delete an input object, if it has been reused
                    if (setGeoList[transClean] != p_geo_in_->getCurrentObject())
                    {
                        delete setGeoList[transClean];
                    }
                }
                delete[] setGeoList;
                return FAIL;
            }
        }
        coDistributedObject *out_set_geo = NULL;
        if (p_create_set_->getValue())
        {
            out_set_geo = new coDoSet(outGeoName.c_str(), setGeoList);
        }
        else
        {
            out_set_geo = AssembleObjects(p_geo_in_->getCurrentObject(),
                                          outGeoName.c_str(), setGeoList);
        }
        for (trans = 0; trans < numTransformations; ++trans)
        {
            // do not delete an input object, if it has been reused
            if (setGeoList[trans] != p_geo_in_->getCurrentObject())
            {
                if (!p_create_set_->getValue() && isUnstructured(setGeoList[trans]))
                {
                    const_cast<coDistributedObject *>(setGeoList[trans])->destroy();
                }
                delete setGeoList[trans];
            }
        }
        delete[] setGeoList;
        p_geo_out_->setCurrentObject(out_set_geo);
    }

    // now process data
    int data_port;
    for (data_port = 0; data_port < NUM_DATA_IN_PORTS; ++data_port)
    {
        const coDistributedObject *dataIn = p_data_in_[data_port]->getCurrentObject();
        if (dataIn == NULL)
        {
            continue;
        }
        std::string outDataName(p_data_out_[data_port]->getObjName());
        if (numTransformations == 1)
        {
            // the last argument means we shall never reuse
            // the input object:
            // possible improvement: !!!!!!!!!!!!!!!!
            // reuse if the input object is already in a set.
            /* FIXME: RedressOrientation might change input objects */
            coDistributedObject *out_data = const_cast<coDistributedObject *>(OutputData(outDataName.c_str(),
                                                                                         dataIn, geometry[0],
                                                                                         h_dataType_[data_port]->getIValue() + 1,
                                                                                         transformations[0],
                                                                                         lagrangeStateTransformations, 0));
            if (out_data == NULL)
            {
                sendError("Could not create output data");
                return FAIL;
            }
            p_data_out_[data_port]->setCurrentObject(out_data);
        }
        else if (numTransformations > 1)
        {
            const coDistributedObject **setDataList = new const coDistributedObject *[numTransformations + 1];
            setDataList[numTransformations] = NULL;
            int trans;
            for (trans = 0; trans < numTransformations; ++trans)
            {
                char bufname[64];
                sprintf(bufname, "%s_%d", outDataName.c_str(), trans);
                if (displacementsPort == data_port
                    && h_type_->getIValue() == TYPE_TILE)
                {
                    // Here the last argument (1) tells OutputData
                    // we are bunching results in a set: we may reuse input objects
                    setDataList[trans] = OutputData(bufname, dataIn, geometry[trans],
                                                    h_dataType_[data_port]->getIValue() + 1,
                                                    transformations[trans],
                                                    lagrangeStateTransformations + trans,
                                                    1);
                }
                else
                {
                    setDataList[trans] = OutputData(bufname, dataIn, geometry[trans],
                                                    h_dataType_[data_port]->getIValue() + 1,
                                                    transformations[trans], NULL, 1);
                }
                if (setDataList[trans] == NULL)
                {
                    sendError("Could not create output data");
                    int transClean;
                    for (transClean = 0; transClean < trans; ++transClean)
                    {
                        if (setDataList[transClean] != dataIn)
                        {
                            delete setDataList[transClean];
                        }
                    }
                    delete[] setDataList;
                    return FAIL;
                }
            }
            coDistributedObject *out_set_data = NULL;
            if (p_create_set_->getValue())
            {
                out_set_data = new coDoSet(outDataName.c_str(), setDataList);
            }
            else
            {
                out_set_data = AssembleObjects(dataIn, outDataName.c_str(), setDataList);
            }
            for (trans = 0; trans < numTransformations; ++trans)
            {
                if (setDataList[trans] != dataIn)
                {
                    if (!p_create_set_->getValue() && isUnstructured(setDataList[trans]))
                    {
                        const_cast<coDistributedObject *>(setDataList[trans])->destroy();
                    }
                    delete setDataList[trans];
                }
            }
            delete[] setDataList;
            /* FIXME: RedressOrientation might change input objects */
            p_data_out_[data_port]->setCurrentObject(out_set_data);
        } // if-then-else on the number of transformations
    } // loop over data ports
    // correct for negative-jacobian transformations
    // in the case of rectilinear and structured grids
    RedressOrientation(numTransformations, transformations);
    delete[] transformations;
    delete[] lagrangeStateTransformations;
    return SUCCESS;
}

void
Transform::RedressOrientation(int numTransformations, Matrix *transformations)
{
    coDistributedObject *outObj = p_geo_out_->getCurrentObject();
    if (outObj == NULL)
        return;

    if (outObj->isType("SETELE"))
    {
        coDoSet *OutObj = (coDoSet *)outObj;
        int no_elements;
        coDistributedObject *const *elemList = const_cast<coDistributedObject *const *>(OutObj->getAllElements(&no_elements));
        if (numTransformations != no_elements)
        {
            return;
        }

        int port;
        for (port = 0; port < NUM_DATA_IN_PORTS; ++port)
        {
            coDistributedObject *outData = p_data_out_[port]->getCurrentObject();
            if (outData == NULL)
            {
                int element;
                for (element = 0; element < no_elements; ++element)
                {
                    RedressOrientation(elemList[element], NULL, port,
                                       transformations + element);
                }
            }
            else if (outData->isType("SETELE"))
            {
                coDoSet *OutData = (coDoSet *)outData;
                int no_d_elements;
                coDistributedObject *const *dataList = const_cast<coDistributedObject *const *>(OutData->getAllElements(&no_d_elements));
                if (no_d_elements == no_elements)
                {
                    int element;
                    for (element = 0; element < no_elements; ++element)
                    {
                        RedressOrientation(elemList[element], dataList[element], port,
                                           transformations + element);
                    }
                }
                int elem;
                for (elem = 0; elem < no_d_elements; ++elem)
                {
                    delete dataList[elem];
                }
                delete[] dataList;
            }
        }
        int elem;
        for (elem = 0; elem < no_elements; ++elem)
        {
            delete elemList[elem];
        }
        delete[] elemList;
    }
    else
    {
        int port;
        for (port = 0; port < NUM_DATA_IN_PORTS; ++port)
        {
            RedressOrientation(outObj, p_data_out_[port]->getCurrentObject(), port,
                               transformations);
        }
    }
}

void
InvertArray(float *data, int len)
{
    int z_level;
    for (z_level = 0; z_level < len / 2; ++z_level)
    {
        float tmp = data[len - 1 - z_level];
        data[len - 1 - z_level] = data[z_level];
        data[z_level] = tmp;
    }
}

void
InvertArray(float *data, int sx, int sy, int sz, bool xInv, bool yInv, bool zInv)
{
    int x, y, z;
    float *tmp = new float[sx * sy * sz];
    for (x = 0; x < sx; ++x)
    {
        int xp = (xInv ? sx - 1 - x : x);
        for (y = 0; y < sy; ++y)
        {
            int yp = (yInv ? sy - 1 - y : y);
            for (z = 0; z < sz; ++z)
            {
                int zp = (zInv ? sz - 1 - z : z);
                int positionp = xp * sy * sz + yp * sz + zp;
                int position = x * sy * sz + y * sz + z;
                tmp[positionp] = data[position];
            }
        }
    }
    memcpy(data, tmp, sizeof(float) * sx * sy * sz);
    delete[] tmp;
}

void
InvertData(coDistributedObject *data, bool xInv, bool yInv, bool zInv, int size[3])
{
    if (data && data->isType("USTSDT"))
    {
        coDoFloat *sdata = (coDoFloat *)(data);
        float *u;
        sdata->getAddress(&u);

        InvertArray(u, size[0], size[1], size[2], xInv, yInv, zInv);
    }
    else if (data && data->isType("USTVDT"))
    {
        coDoVec3 *vdata = (coDoVec3 *)(data);
        float *u[3];
        vdata->getAddresses(&u[0], &u[1], &u[2]);

        InvertArray(u[0], size[0], size[1], size[2], xInv, yInv, zInv);
        InvertArray(u[1], size[0], size[1], size[2], xInv, yInv, zInv);
        InvertArray(u[2], size[0], size[1], size[2], xInv, yInv, zInv);
    }
    return;
}

void
Transform::RedressOrientation(coDistributedObject *grid,
                              coDistributedObject *data,
                              int port,
                              Matrix *transformation)
{
    if (grid == NULL || transformation == NULL)
    {
        return;
    }
    // STRGRD
    int sx = -1, sy = -1, sz = -1;
    if (grid->isType("STRGRD") || grid->isType("UNIGRD") || grid->isType("RCTGRD"))
    {
        coDoAbstractStructuredGrid *strgrid = (coDoAbstractStructuredGrid *)(grid);
        strgrid->getGridSize(&sx, &sy, &sz);
    }
    int sizes[3] = { sx, sy, sz };

    if (grid->isType("STRGRD") && port == 0 && transformation->getJacobian() == Matrix::NEG_JACOBIAN)
    {
        coDoStructuredGrid *strgrid = (coDoStructuredGrid *)(grid);
        float *xc, *yc, *zc;
        strgrid->getAddresses(&xc, &yc, &zc);

        int z_column;
        for (z_column = 0; z_column < sx * sy; ++z_column)
        {
            float *u[3];
            u[0] = xc;
            u[1] = yc;
            u[2] = zc;
            int dim;
            for (dim = 0; dim < 3; ++dim)
            {
                float *base = u[dim] + sz * z_column;
                InvertArray(base, sz);
            }
        }
    }
    if (grid->isType("STRGRD") && transformation->getJacobian() == Matrix::NEG_JACOBIAN
        && data != NULL && data->isType("USTSDT"))
    {
        coDoFloat *sdata = (coDoFloat *)(data);
        float *u;
        sdata->getAddress(&u);

        int z_column;
        for (z_column = 0; z_column < sx * sy; ++z_column)
        {
            float *base = u + sz * z_column;
            InvertArray(base, sz);
        }
    }
    if (grid->isType("STRGRD") && transformation->getJacobian() == Matrix::NEG_JACOBIAN
        && data != NULL && data->isType("USTVDT"))
    {
        coDoVec3 *vdata = (coDoVec3 *)(data);
        float *u[3];
        vdata->getAddresses(&u[0], &u[1], &u[2]);

        int z_column;
        for (z_column = 0; z_column < sx * sy; ++z_column)
        {
            int dim;
            for (dim = 0; dim < 3; ++dim)
            {
                float *base = u[dim] + sz * z_column;
                InvertArray(base, sz);
            }
        }
    }
    // RCTGRD: this is even more complicated...
    if (grid->isType("RCTGRD"))
    {
        coDoRectilinearGrid *rctgrid = (coDoRectilinearGrid *)(grid);
        float *xc, *yc, *zc;
        rctgrid->getAddresses(&xc, &yc, &zc);
        // we have to check whether some directions have to be inverted
        bool xInv = false, yInv = false, zInv = false;
        if (xc[0] > xc[sx - 1])
        {
            xInv = true;
        }
        if (yc[0] > yc[sy - 1])
        {
            yInv = true;
        }
        if (zc[0] > zc[sz - 1])
        {
            zInv = true;
        }
        // invert geometric grid arrays only when processing the last port!!!
        // otherwise only the first data port would be corrected if necessary
        if (port == NUM_DATA_IN_PORTS - 1)
        {
            if (xInv)
            {
                InvertArray(xc, sx);
            }
            if (yInv)
            {
                InvertArray(yc, sy);
            }
            if (zInv)
            {
                InvertArray(zc, sz);
            }
        }
        InvertData(data, xInv, yInv, zInv, sizes);
        /*
            if(data && data->isType("USTSDT")){
               coDoFloat *sdata =
                  (coDoFloat *)(data);
               float *u;
               sdata->getAddress(&u);
               int sx,sy,sz;
               sdata->getGridSize(&sx,&sy,&sz);

               InvertArray(u,sx,sy,sz,xInv,yInv,zInv);
            }
      else if(data && data->isType("USTVDT")){
      coDoVec3 *vdata =
      (coDoVec3 *)(data);
      float *u[3];
      vdata->getAddresses(&u[0],&u[1],&u[2]);
      int sx,sy,sz;
      vdata->getGridSize(&sx,&sy,&sz);

      InvertArray(u[0],sx,sy,sz,xInv,yInv,zInv);
      InvertArray(u[1],sx,sy,sz,xInv,yInv,zInv);
      InvertArray(u[2],sx,sy,sz,xInv,yInv,zInv);
      }
      */
    }
    // UNIGRD:
    if (grid->isType("UNIGRD"))
    {
        coDoUniformGrid *unigrid = (coDoUniformGrid *)(grid);
        float xc[2], yc[2], zc[2];
        unigrid->getMinMax(xc, xc + 1, yc, yc + 1, zc, zc + 1);
        // we have to check whether some directions have to be inverted
        bool xInv = false, yInv = false, zInv = false;
        if (xc[0] > xc[1])
        {
            xInv = true;
        }
        if (yc[0] > yc[1])
        {
            yInv = true;
        }
        if (zc[0] > zc[1])
        {
            zInv = true;
        }
        // invert geometric grid arrays only when processing the last port!!!
        // otherwise only the first data port would be corrected if necessary
        if (port == NUM_DATA_IN_PORTS - 1)
        {
            if (xInv)
            {
                unigrid->SwapMinMax(0); //InvertArray(xc,2);
            }
            if (yInv)
            {
                unigrid->SwapMinMax(1); //InvertArray(yc,2);
            }
            if (zInv)
            {
                unigrid->SwapMinMax(2); //InvertArray(zc,2);
            }
        }
        InvertData(data, xInv, yInv, zInv, sizes);
        /*
            if(data && data->isType("USTSDT")){
               coDoFloat *sdata =
                  (coDoFloat *)(data);
               float *u;
               sdata->getAddress(&u);
               int sx,sy,sz;
               sdata->getGridSize(&sx,&sy,&sz);

               InvertArray(u,sx,sy,sz,xInv,yInv,zInv);
            }
      else if(data && data->isType("USTVDT")){
      coDoVec3 *vdata =
      (coDoVec3 *)(data);
      float *u[3];
      vdata->getAddresses(&u[0],&u[1],&u[2]);
      int sx,sy,sz;
      vdata->getGridSize(&sx,&sy,&sz);

      InvertArray(u[0],sx,sy,sz,xInv,yInv,zInv);
      InvertArray(u[1],sx,sy,sz,xInv,yInv,zInv);
      InvertArray(u[2],sx,sy,sz,xInv,yInv,zInv);
      }
      */
    }
}

MODULE_MAIN(Tools, Transform)
