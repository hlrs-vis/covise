#ifndef COGRMSG_DECL_H
#define COGRMSG_DECL_H

/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifndef COGRMSG_H
#define COGRMSG_H

#include <string>
#include <vector>
#include <util/coExport.h>
#include <memory>
#ifdef WIN32
#define strdup _strdup
#endif //WIN32
#ifndef COGRMSG_SAFEFREE
#define COGRMSG_SAFEFREE(x) \
    if (x)                  \
    {                       \
        free(x);            \
        x = NULL;           \
    }
#endif //COGRMSG_SAFEFREE

namespace grmsg
{

class GRMSGEXPORT coGRMsg
{
public:
    /// id to specify the type of the message
    typedef enum
    {
        NO_TYPE = 0,
        REGISTER,
        GEO_VISIBLE,
        INTERACTOR_VISIBLE,
        SMOKE_VISIBLE,
        MOVE_INTERACTOR,
        INTERACTOR_USED,
        CREATE_VIEWPOINT,
        CREATE_DEFAULT_VIEWPOINT,
        SHOW_VIEWPOINT, //9
        SHOW_PRESENTATIONPOINT,
        CHANGE_VIEWPOINT_ID,
        CHANGE_VIEWPOINT_NAME,
        DELETE_VIEWPOINT,
        SET_VIEWPOINT_FILE,
        FLYMODE_TOGGLE,
        SET_CASE,
        SET_NAME,
        MOVE_OBJECT,
        ADD_DOCUMENT,
        SET_DOCUMENT_PAGE, //20
        SET_DOCUMENT_SCALE,
        SET_DOCUMENT_POSITION,
        SET_DOCUMENT_PAGESIZE,
        SEND_DOCUMENT_NUMBERS,
        DOC_VISIBLE,
        BOUNDARIES_OBJECT,
        COLOR_OBJECT,
        SHADER_OBJECT, // 28
        MATERIAL_OBJECT,
        SET_TRANSPARENCY,
        KEYWORD,//31
        TRANSFORM_OBJECT,
        TRANSFORM_CASE,
        RESTRICT_AXIS,
        GRAPHIC_RESSOURCE,
        SET_MOVE,
        SET_MOVE_SELECTED,
        SENSOR,
        SENSOR_EVENT,
        ATTACHED_CLIPPLANE,
        ANIMATION_ON,//41
        ANIMATION_SPEED,
        ANIMATION_TIMESTEP,
        ACTIVATED_VIEWPOINT,
        VPCLIPPLANEMODE_TOGGLE,
        SNAPSHOT,//46
        SEND_CURRENT_DOCUMENT,
        SET_TRACKING_PARAMS,
        CHANGE_VIEWPOINT,
        VIEWPOINT_CHANGED,
        OBJECT_TRANSFORMED,
        SET_CONNECTIONPOINT,
        GENERIC_PARAM_REGISTER,
        GENERIC_PARAM_CHANGED,
        TRANSFORM_SGITEM,
        SELECT_OBJECT,
        DELETE_OBJECT,
        GEOMETRY_OBJECT, // KLSM
        ADD_CHILD_OBJECT, // KLSM
        TURNTABLE_ANIMATION,
        SET_VARIANT, // KLSM
        SET_APPEARANCE, // KLSM
        KINEMATICS_STATE, // KLSM
        PLUGIN
    } Mtype; //, PLANEINT, BBINT, COLOR, MATERIAL, SELECT, VIEWPOINT, CLIPLANE } Mtype;

    /// recreate class from a message string
    coGRMsg(const char *msg);
    template <typename T>
    const T &as() const { return dynamic_cast<const T&>(*this); }
    /// destructor
    virtual ~coGRMsg() = default;

    /// whether recreate was succesful
    int isValid() const
    {
        return (is_valid_ == 1);
    };

    /// output content to stdout
    virtual void print_stdout();

    /// access to private variables
    Mtype getType() const
    {
        return type_;
    };
    const char *c_str() const;
    std::string getString() const;

protected:
    /// construct used by child class
    coGRMsg(Mtype type);

    /// add a token to the message
    void addToken(const char *token);

    /// read first token in the message
    std::string getFirstToken();

    /// read and delete first token in the message
    std::string extractFirstToken();

    /// read all tokens at once
    std::vector<std::string> getAllTokens();

    /// valid can be overwritten by children if their recreation was not succesful
    int is_valid_;

private:
    Mtype type_ = NO_TYPE;
    std::string content_;
};

}
#endif


#endif // COGRMSG_DECL_H