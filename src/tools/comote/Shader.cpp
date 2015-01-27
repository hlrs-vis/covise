/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** \file
 * \brief Ease using GLSL shaders.
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <GL/glew.h>
#include "Shader.h"
#include <iostream>
#include <QDebug>
#include <QString>
#include <QFile>

#ifdef SOLUTION
#include "../../obscure/obscure.h"
#endif

Shader::Shader()
    : m_programId(0)
{
}

Shader::~Shader()
{
    if (haveGLSL())
    {
        glDeleteProgram(m_programId);
        m_programId = 0;
    }
}

QString
Shader::loadFile(QString filename)
{
    QFile file(filename);
    QString content;
    if (file.open(QIODevice::ReadOnly))
        content = QString(file.readAll());
    else
        qDebug() << "could not open" << filename;
#ifdef SOLUTION
    if (filename.endsWith('o'))
        return obscure(content);
    else
        return content;
#else
    return content;
#endif
}

bool
Shader::loadVertexSource(QString filename)
{
    QString prog = loadFile(filename);
    bool ok = !prog.isEmpty();
    if (ok)
        m_vertexSource = prog;
    else
        m_vertexSource.clear();
    return ok;
}

bool
Shader::loadFragmentSource(QString filename)
{
    QString prog = loadFile(filename);
    bool ok = !prog.isEmpty();
    if (ok)
        m_fragmentSource = prog;
    else
        m_fragmentSource.clear();
    return ok;
}

void
printGLError(const char *msg)
{
    while (GLenum err = glGetError())
    {
        const GLubyte *str = gluErrorString(err);
        qDebug() << "GL error:" << msg << "-" << str;
    }
}

bool
Shader::printShaderLog(GLuint id, bool vertex)
{
    GLint logLen;
    GLchar *infoLog;

    printGLError("shader log");
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0)
    {
        infoLog = new GLchar[logLen];
        glGetShaderInfoLog(id, logLen, NULL, infoLog);
        if (strlen(infoLog) > 0)
        {
            qDebug() << (vertex ? "Vertex" : "Fragment") << "shader log:" << infoLog;
            delete[] infoLog;
            return false;
        }
        delete[] infoLog;
    }

    return true;
}

bool
Shader::printProgramLog(GLuint id)
{
    GLint logLen;
    GLchar *infoLog;

    printGLError("program log");
    glGetProgramiv(id, GL_INFO_LOG_LENGTH, &logLen);
    printGLError("program log 2");
    if (logLen > 0)
    {
        infoLog = new GLchar[logLen];
        glGetProgramInfoLog(id, logLen, NULL, infoLog);
        if (strlen(infoLog) > 0)
        {
            qDebug() << "Program log:" << infoLog;
            delete[] infoLog;
            return false;
        }
        delete[] infoLog;
    }
    return true;
}

bool
Shader::link()
{
    if (!haveGLSL())
        return false;

    if (m_programId)
        glDeleteProgram(m_programId);
    m_programId = 0;

    QByteArray vert = m_vertexSource.toLocal8Bit();
    const char *vs = vert.constData();
    GLuint vertexId = 0;
    if (vs && strcmp(vs, ""))
    {
        vertexId = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexId, 1, &vs, NULL);
        glCompileShader(vertexId);
        printShaderLog(vertexId, true);
    }

    QByteArray frag = m_fragmentSource.toLocal8Bit();
    const char *fs = frag.constData();
    GLuint fragmentId = 0;
    if (fs && strcmp(fs, ""))
    {
        fragmentId = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentId, 1, &fs, NULL);
        glCompileShader(fragmentId);
        printShaderLog(fragmentId, false);
    }

    m_programId = glCreateProgram();
    if (vertexId)
        glAttachShader(m_programId, vertexId);
    if (fragmentId)
        glAttachShader(m_programId, fragmentId);
    glLinkProgram(m_programId);
    printProgramLog(m_programId);
    if (vertexId)
        glDeleteShader(vertexId);
    if (fragmentId)
        glDeleteShader(fragmentId);

    return true;
}

void
Shader::disable()
{
    if (haveGLSL())
    {
        glUseProgram(0);
    }
}

void
Shader::enable()
{
    if (haveGLSL())
    {
        printGLError("trying to validate program");

        if (m_programId)
        {
            glValidateProgram(m_programId);
            printGLError("validate program");
            GLint valid;
            glGetProgramiv(m_programId, GL_VALIDATE_STATUS, &valid);
            if (valid == GL_FALSE)
            {
                printProgramLog(m_programId);
            }
        }
        glUseProgram(m_programId);
        printGLError("use program");
    }
}

bool
Shader::setUniform1f(const char *name, float value)
{
    GLint loc = glGetUniformLocation(m_programId, name);
    if (loc == -1)
        return false;

    glUniform1f(loc, value);
    return true;
}

bool
Shader::setUniform1i(const char *name, int value)
{
    GLint loc = glGetUniformLocation(m_programId, name);
    if (loc == -1)
        return false;

    glUniform1i(loc, value);
    return true;
}

bool
Shader::haveGLSL()
{
    return glewIsSupported("GL_VERSION_2_0");
}
