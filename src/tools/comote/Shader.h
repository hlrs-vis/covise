/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** \file
 * \brief Ease using GLSL shaders.
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>

#include <QString>

//! A simple class to enable and disable shader programs
/*! \anchor Shader
 * \b Shader is provides the ability to enable and disable vertex and
 * fragment processing using the programmable OpenGL pipeline
 */
class Shader
{
public:
    //! default constructor
    Shader();
    //! destructor
    virtual ~Shader();

    //! returns true iff the programmable OpenGL (aka OpenGL 2.0) pipeline is available
    static bool haveGLSL();

    //! load the source for the vertex program from a file
    bool loadVertexSource(QString filename);
    //! load the source for the fragment program from a file
    bool loadFragmentSource(QString filename);

    //! compile the vertex and fragment sources and link them to a program
    bool link();

    //! re-enable the fixed function OpenGL processing
    void disable();
    //! use the shader
    void enable();

    //! set a scalar uniform variable
    bool setUniform1f(const char *name, float value);

    //! set an integer uniform variable
    bool setUniform1i(const char *name, int value);

private:
    //! id as returned by glCreateProgram
    GLuint m_programId;
    //! internal routine for loading files
    QString loadFile(QString filename);

    //! the source for the vertex shader
    QString m_vertexSource;
    //! the source for the fragment shader
    QString m_fragmentSource;

    //!
    bool printShaderLog(GLuint id, bool vertex);
    //!
    bool printProgramLog(GLuint id);
};
#endif
