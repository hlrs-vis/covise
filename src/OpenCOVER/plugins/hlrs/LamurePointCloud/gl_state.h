#ifndef GL_STATE_H
#define GL_STATE_H

#ifndef __gl_h_
#include <GL/glew.h>
#endif

#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <algorithm>
#include <windows.h>
#include <sstream>

// -----------------------------------------------------------------------------
//  Einzelner Buffer‑Zustand
// -----------------------------------------------------------------------------
struct GLBufferInfo {
    GLuint id;              // Buffer‑Handle
    GLint  size;            // Größe in Bytes
    std::string target;     // z.B. "GL_ARRAY_BUFFER"
};

// -----------------------------------------------------------------------------
//  Einzelner Vertex‑Attrib‑Zustand
// -----------------------------------------------------------------------------
struct GLVertexAttribInfo {
    GLint     index;        // Attribut‑Index
    GLboolean enabled;      // aktiviert?
    GLint     size;         // Komponenten‑Anzahl
    GLenum    type;         // z.B. GL_FLOAT
    GLboolean normalized;   // normalisiert?
    GLint     stride;       // Byte‑Offset zwischen Elementen
    GLint     bufferBinding;// aktuell gebundener Buffer
    void*     pointer;      // Vertex‑Pointer
    GLint     divisor;      // Instancing‑Divisor
};



// Struktur für ein Attachment
struct GLFBOAttachmentInfo {
    GLenum attachmentPoint;       // z.B. GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, …
    std::string objectType;       // "NONE", "TEXTURE", "RENDERBUFFER"
    GLuint      objectName;       // Objekt-ID
    GLenum      internalFormat;   // nur für TEXTURE, sonst 0
};

// Gesamt-Info zum Framebuffer
struct GLFramebufferInfo {
    GLuint                          fboBinding;
    std::string                     completeness;   // z.B. "COMPLETE", "INCOMPLETE_ATTACHMENT", …
    GLenum                          readBuffer;
    std::vector<GLenum>             drawBuffers;
    std::vector<GLFBOAttachmentInfo> attachments;
};

// -----------------------------------------------------------------------------
//  Snapshot aller relevanten OpenGL‑Zustände
// -----------------------------------------------------------------------------
class GLState {
public:
    std::set<GLuint> enumerateAllBufferIDs(GLuint maxID);
    static std::string fboStatusToString(GLenum status);
    static std::vector<GLVertexAttribInfo> captureVertexAttribState();
    static GLFramebufferInfo queryCurrentFramebuffer(GLint maxColorAttachments = 8);
    static void printFramebufferInfo(const GLFramebufferInfo& info);
    //--------------------------------------------------------------------------
    // 1) Erstelle ein "Snapshot" aller States
    //--------------------------------------------------------------------------
    static GLState capture();

    //--------------------------------------------------------------------------
    // 2) Stelle genau die States wieder her, die in dieser Instanz gespeichert sind
    //--------------------------------------------------------------------------
    void restore() const;

    //--------------------------------------------------------------------------
    // Helper zum Aktivieren / Deaktivieren von Capabilities
    //--------------------------------------------------------------------------
    void setEnable(GLenum cap, GLboolean enabled) const;

    //--------------------------------------------------------------------------
    // 3) Vergleiche zwei Snapshots und schreib Unterschiede auf std::cout
    //--------------------------------------------------------------------------
    static void compare(const GLState& before, const GLState& after,
        const std::string& info);

    // Utility‑Funktionen ------------------------------------------------------
    static void printAllExistingBuffers(GLuint maxID);
    static void printCurrentContext();
    static void printBufferContents(GLenum target, GLuint buffer, size_t sizeInBytes);
    static void printVAOAttributes(GLuint vao);

    // Getter -----------------------------------------------------------------
    GLint getVertexArrayBinding() const { return m_vertexArrayBinding; }

private:
    static void printCandidateBuffer(GLuint candidate,
        const std::vector<GLenum>& targets);

    // ------------------------------------------------------------------------
    //  Basis‑Zustände
    // ------------------------------------------------------------------------
    GLint       m_currentProgram;
    GLint       m_vertexArrayBinding;
    GLint       m_arrayBufferBinding;
    GLint       m_elementArrayBufferBinding;
    GLint       m_arrayBufferSize;
    GLint       m_elementArrayBufferSize;

    // Enable‑Flags -----------------------------------------------------------
    GLboolean   m_cullFaceEnabled;
    GLboolean   m_depthTestEnabled;
    GLboolean   m_blendEnabled;
    GLboolean   m_polyOffsetFillEnabled;   // NEU
    GLboolean   m_polyOffsetLineEnabled;   // NEU

    // Rasterizer -------------------------------------------------------------
    GLint       m_polygonMode[2];          // Front / Back
    GLint       m_cullFaceMode;            // NEU
    GLint       m_frontFace;               // NEU
    GLfloat     m_lineWidth;
    GLfloat     m_pointSize;
    GLfloat     m_polygonOffsetFactor;
    GLfloat     m_polygonOffsetUnits;

    // Depth ------------------------------------------------------------------
    GLboolean   m_depthMask;               // NEU
    GLint       m_depthFunc;               // NEU
    GLfloat     m_depthRange[2];
    GLfloat     m_clearDepth;

    // Blend ------------------------------------------------------------------
    GLfloat     m_blendColor[4];
    GLint       m_blendSrcRGB;
    GLint       m_blendDstRGB;
    GLint       m_blendSrcAlpha;
    GLint       m_blendDstAlpha;
    GLint       m_blendEquationRGB;
    GLint       m_blendEquationAlpha;

    // Viewport / Scissor / Clear --------------------------------------------
    GLint       m_viewport[4];
    GLint       m_scissorBox[4];
    GLfloat     m_clearColor[4];

    // Textur‑ / FBO‑Bindungen -----------------------------------------------
    GLint       m_activeTexture;
    GLint       m_textureBinding2D;
    GLint       m_framebufferBinding;
    GLint       m_renderbufferBinding;

    // Buffer‑ und Attrib‑Listen ---------------------------------------------
    std::vector<GLBufferInfo>        m_bufferInfos;
    std::vector<GLVertexAttribInfo>  m_vertexAttribInfos;
};

#endif // GL_STATE_H
