#include "gl_state.h"

// Hilfsfunktion: Alle existierenden Buffer-IDs bis maxID ermitteln
std::set<GLuint> GLState::enumerateAllBufferIDs(GLuint maxID = 100) {
    std::set<GLuint> buffers;
    for (GLuint id = 1; id <= maxID; ++id) {
        if (glIsBuffer(id))
            buffers.insert(id);
    }
    return buffers;
}

// Hilfsfunktion: Informationen zu einem Buffer abfragen
static GLBufferInfo queryBufferInfo(GLuint id, GLenum target) {
    GLBufferInfo info;
    info.id = id;
    info.target = (target == GL_ARRAY_BUFFER)
        ? "GL_ARRAY_BUFFER"
        : "GL_ELEMENT_ARRAY_BUFFER";

    GLint oldBinding = 0;
    if (target == GL_ARRAY_BUFFER)
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBinding);
    else
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &oldBinding);

    glBindBuffer(target, id);
    glGetBufferParameteriv(target, GL_BUFFER_SIZE, &info.size);
    glBindBuffer(target, oldBinding);
    return info;
}

// Hilfsfunktion: Zustand aller Vertex-Attribs erfassen
std::vector<GLVertexAttribInfo> GLState::captureVertexAttribState() {
    std::vector<GLVertexAttribInfo> attribs;
    GLint maxAttribs = 0;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxAttribs);

    for (GLint i = 0; i < maxAttribs; ++i) {
        GLVertexAttribInfo a;
        a.index = i;
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, reinterpret_cast<GLint*>(&a.enabled));
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_SIZE, &a.size);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_TYPE, reinterpret_cast<GLint*>(&a.type));
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, reinterpret_cast<GLint*>(&a.normalized));
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &a.stride);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &a.bufferBinding);
        glGetVertexAttribPointerv(i, GL_VERTEX_ATTRIB_ARRAY_POINTER, &a.pointer);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &a.divisor);
        attribs.push_back(a);
    }
    return attribs;
}

// Hilfsfunktion: Status-String aus glCheckFramebufferStatus
std::string GLState::fboStatusToString(GLenum status) {
    switch (status) {
    case GL_FRAMEBUFFER_COMPLETE:                      return "COMPLETE";
    case GL_FRAMEBUFFER_UNDEFINED:                     return "UNDEFINED";
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:         return "INCOMPLETE_ATTACHMENT";
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return "MISSING_ATTACHMENT";
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:        return "INCOMPLETE_DRAW_BUFFER";
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:        return "INCOMPLETE_READ_BUFFER";
    case GL_FRAMEBUFFER_UNSUPPORTED:                   return "UNSUPPORTED";
    default: {
        std::ostringstream ss;
        ss << "UNKNOWN(0x" << std::hex << status << std::dec << ")";
        return ss.str();
    }
    }
}

// Die Hauptfunktion: FBO-Zustand abfragen
GLFramebufferInfo GLState::queryCurrentFramebuffer(GLint maxColorAttachments) {
    GLFramebufferInfo info;

    // 1) Welches FBO ist gebunden?
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&info.fboBinding));

    // 2) Vollständigkeits-Status
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    info.completeness = fboStatusToString(status);

    // 3) Read-Buffer
    glGetIntegerv(GL_READ_BUFFER, reinterpret_cast<GLint*>(&info.readBuffer));

    // 4) Draw-Buffers
    GLint maxDraw = 0;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxDraw);
    maxDraw = std::min<GLint>(maxDraw, maxColorAttachments);
    info.drawBuffers.resize(maxDraw);
    for (GLint i = 0; i < maxDraw; ++i) {
        glGetIntegerv(GL_DRAW_BUFFER0 + i, reinterpret_cast<GLint*>(&info.drawBuffers[i]));
    }

    // 5) Attachments abfragen: Color0…ColorN, Depth, Stencil
    auto queryAttachment = [&](GLenum attachmentPoint){
        GLFBOAttachmentInfo ai;
        ai.attachmentPoint = attachmentPoint;
        ai.internalFormat = 0;

        GLint objType = 0, objName = 0;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,
            attachmentPoint,
            GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
            &objType);
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,
            attachmentPoint,
            GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
            &objName);

        if (objType == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) {
            ai.objectType = "NONE";
        }
        else if (objType == GL_TEXTURE) {
            ai.objectType = "TEXTURE";
            ai.objectName = objName;
            // internen Format auslesen
            GLenum prevTex = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&prevTex));
            glBindTexture(GL_TEXTURE_2D, objName);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                GL_TEXTURE_INTERNAL_FORMAT,
                reinterpret_cast<GLint*>(&ai.internalFormat));
            glBindTexture(GL_TEXTURE_2D, prevTex);
        }
        else if (objType == GL_RENDERBUFFER) {
            ai.objectType = "RENDERBUFFER";
            ai.objectName = objName;
        }
        else {
            ai.objectType = "UNKNOWN";
            ai.objectName = objName;
        }

        return ai;
        };

    // Color-Anhänge
    for (GLint i = 0; i < maxColorAttachments; ++i) {
        info.attachments.push_back(
            queryAttachment(GL_COLOR_ATTACHMENT0 + i));
    }
    // Depth- und Stencil-Anhang
    info.attachments.push_back(queryAttachment(GL_DEPTH_ATTACHMENT));
    info.attachments.push_back(queryAttachment(GL_STENCIL_ATTACHMENT));

    return info;
}

// Debug-Ausgabe
void GLState::printFramebufferInfo(const GLFramebufferInfo &info) {
    std::cout << "=== Framebuffer ID: " << info.fboBinding << " ===\n";
    std::cout << "Status: " << info.completeness << "\n";
    std::cout << "Read Buffer: 0x" << std::hex << info.readBuffer << std::dec << "\n";
    for (size_t i = 0; i < info.drawBuffers.size(); ++i) {
        std::cout << "DrawBuffer["<<i<<"]: 0x" << std::hex << info.drawBuffers[i] << std::dec << "\n";
    }
    for (auto &a : info.attachments) {
        std::cout << "Attachment 0x" << std::hex << a.attachmentPoint << std::dec
            << " Type=" << a.objectType
            << " Name=" << a.objectName;
        if (a.internalFormat != 0) {
            std::cout << " InternalFormat=0x" << std::hex << a.internalFormat << std::dec;
        }
        std::cout << "\n";
    }
    std::cout << "===============================\n";
}

// ----- 1) Capture aller GL-States -----
GLState GLState::capture()
{
    GLState s;

    // --- Basis-Bindungen ----------------------------------------------------
    glGetIntegerv(GL_CURRENT_PROGRAM,               &s.m_currentProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,          &s.m_vertexArrayBinding);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,          &s.m_arrayBufferBinding);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING,  &s.m_elementArrayBufferBinding);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING,           &s.m_framebufferBinding);
    glGetIntegerv(GL_RENDERBUFFER_BINDING,          &s.m_renderbufferBinding);
    glGetIntegerv(GL_ACTIVE_TEXTURE,                &s.m_activeTexture);
    glGetIntegerv(GL_TEXTURE_BINDING_2D,            &s.m_textureBinding2D);

    // --- Enable-Flags -------------------------------------------------------
    s.m_cullFaceEnabled        = glIsEnabled(GL_CULL_FACE);
    s.m_depthTestEnabled       = glIsEnabled(GL_DEPTH_TEST);
    s.m_blendEnabled           = glIsEnabled(GL_BLEND);
    s.m_polyOffsetFillEnabled  = glIsEnabled(GL_POLYGON_OFFSET_FILL);
    s.m_polyOffsetLineEnabled  = glIsEnabled(GL_POLYGON_OFFSET_LINE);

    // --- Rasterizer ---------------------------------------------------------
    glGetIntegerv(GL_CULL_FACE_MODE,   &s.m_cullFaceMode);
    glGetIntegerv(GL_FRONT_FACE,       &s.m_frontFace);
    glGetIntegerv(GL_POLYGON_MODE,      s.m_polygonMode);   // [0]=Front,[1]=Back
    glGetFloatv (GL_LINE_WIDTH,        &s.m_lineWidth);
    glGetFloatv (GL_POINT_SIZE,        &s.m_pointSize);
    glGetFloatv (GL_POLYGON_OFFSET_FACTOR, &s.m_polygonOffsetFactor);
    glGetFloatv (GL_POLYGON_OFFSET_UNITS,  &s.m_polygonOffsetUnits);

    // --- Depth --------------------------------------------------------------
    glGetIntegerv(GL_DEPTH_FUNC,       &s.m_depthFunc);
    glGetBooleanv(GL_DEPTH_WRITEMASK,  &s.m_depthMask);
    glGetFloatv (GL_DEPTH_CLEAR_VALUE, &s.m_clearDepth);
    glGetFloatv (GL_DEPTH_RANGE,        s.m_depthRange);    // [0]=near,[1]=far

    // --- Blend --------------------------------------------------------------
    glGetFloatv (GL_BLEND_COLOR,            s.m_blendColor);
    glGetIntegerv(GL_BLEND_SRC_RGB,        &s.m_blendSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB,        &s.m_blendDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,      &s.m_blendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA,      &s.m_blendDstAlpha);
    glGetIntegerv(GL_BLEND_EQUATION_RGB,   &s.m_blendEquationRGB);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &s.m_blendEquationAlpha);

    // --- Viewport / Scissor / Clear ----------------------------------------
    glGetIntegerv(GL_VIEWPORT,     s.m_viewport);
    glGetIntegerv(GL_SCISSOR_BOX,  s.m_scissorBox);
    glGetFloatv (GL_COLOR_CLEAR_VALUE, s.m_clearColor);

    // --- Vertex-Attrib Arrays ----------------------------------------------
    s.m_vertexAttribInfos = captureVertexAttribState();

    // --- Puffer-Größen (robust, 64-Bit) ------------------------------------
    // ELEMENT_ARRAY_BUFFER ist VAO-lokal: zum Abfragen muss das passende VAO gebunden sein.
    auto getBufSize64 = [](GLenum target) -> GLint64 {
        GLint64 sz = 0;
        glGetBufferParameteri64v(target, GL_BUFFER_SIZE, &sz);
        return sz;
        };

    // Ursprungs-Bindings merken
    GLint prevVAO   = 0;
    GLint prevArray = 0;
    GLint prevElem  = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,         &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,         &prevArray);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElem);

    // ARRAY_BUFFER (global, nicht VAO-lokal)
    if (s.m_arrayBufferBinding != 0) {
        glBindBuffer(GL_ARRAY_BUFFER, s.m_arrayBufferBinding);
        s.m_arrayBufferSize = getBufSize64(GL_ARRAY_BUFFER);
    } else {
        s.m_arrayBufferSize = 0;
    }

    // ELEMENT_ARRAY_BUFFER (VAO-lokal)
    if (s.m_vertexArrayBinding != 0) {
        glBindVertexArray(s.m_vertexArrayBinding);

        GLint ebo = 0;
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
        if (ebo != 0) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
            s.m_elementArrayBufferSize = getBufSize64(GL_ELEMENT_ARRAY_BUFFER);
        } else {
            s.m_elementArrayBufferSize = 0;
        }
    } else {
        s.m_elementArrayBufferSize = 0;
    }

    // Ursprungs-Bindings wiederherstellen
    glBindBuffer(GL_ARRAY_BUFFER,         prevArray);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElem);
    glBindVertexArray(prevVAO);

    return s;
}


void GLState::restore() const
{
    // --- Bindings -----------------------------------------------------------
    glUseProgram       (m_currentProgram);
    glBindVertexArray  (m_vertexArrayBinding);
    glBindFramebuffer  (GL_FRAMEBUFFER,  m_framebufferBinding);
    glBindRenderbuffer (GL_RENDERBUFFER, m_renderbufferBinding);
    glBindBuffer       (GL_ARRAY_BUFFER, m_arrayBufferBinding);
    glBindBuffer       (GL_ELEMENT_ARRAY_BUFFER, m_elementArrayBufferBinding);

    glActiveTexture    (m_activeTexture);
    glBindTexture      (GL_TEXTURE_2D, m_textureBinding2D);

    // --- Enable-Flags -------------------------------------------------------
    setEnable(GL_CULL_FACE,           m_cullFaceEnabled);
    setEnable(GL_DEPTH_TEST,          m_depthTestEnabled);
    setEnable(GL_BLEND,               m_blendEnabled);
    setEnable(GL_POLYGON_OFFSET_FILL, m_polyOffsetFillEnabled);
    setEnable(GL_POLYGON_OFFSET_LINE, m_polyOffsetLineEnabled);

    // --- Rasterizer ---------------------------------------------------------
    glCullFace   (m_cullFaceMode);
    glFrontFace  (m_frontFace);
    glPolygonMode(GL_FRONT, m_polygonMode[0]);
    glPolygonMode(GL_BACK,  m_polygonMode[1]);
    glLineWidth  (m_lineWidth);
    glPointSize  (m_pointSize);
    glPolygonOffset(m_polygonOffsetFactor, m_polygonOffsetUnits);

    // --- Depth --------------------------------------------------------------
    glDepthFunc (m_depthFunc);
    glDepthMask (m_depthMask);
    glClearDepth(m_clearDepth);
    glDepthRangef(m_depthRange[0], m_depthRange[1]);

    // --- Blend --------------------------------------------------------------
    glBlendColor (m_blendColor[0], m_blendColor[1], m_blendColor[2], m_blendColor[3]);
    glBlendFuncSeparate(m_blendSrcRGB,  m_blendDstRGB,
        m_blendSrcAlpha,m_blendDstAlpha);
    glBlendEquationSeparate(m_blendEquationRGB, m_blendEquationAlpha);

    // --- Viewport / Scissor / Clear ----------------------------------------
    glViewport (m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
    glScissor  (m_scissorBox[0], m_scissorBox[1],
        m_scissorBox[2], m_scissorBox[3]);
    glClearColor(m_clearColor[0], m_clearColor[1],
        m_clearColor[2], m_clearColor[3]);

    // --- Vertex-Attrib Arrays ----------------------------------------------
    for (const auto& a : m_vertexAttribInfos) {
        glBindBuffer(GL_ARRAY_BUFFER, a.bufferBinding);
        glVertexAttribPointer(a.index, a.size, a.type,
            a.normalized, a.stride, a.pointer);
        glVertexAttribDivisor(a.index, a.divisor);
        if (a.enabled) glEnableVertexAttribArray(a.index);
        else           glDisableVertexAttribArray(a.index);
    }

    // Wichtig: VBO-Bindung auf zuletzt gültigen Wert zurücksetzen
    glBindBuffer(GL_ARRAY_BUFFER, m_arrayBufferBinding);
}

inline void GLState::setEnable(GLenum cap, GLboolean enabled) const
{
    if (enabled) glEnable(cap);
    else         glDisable(cap);
}


void GLState::compare(const GLState& before,
    const GLState& after,
    const std::string& info)
{
    std::ostringstream log;
    bool any = false;

    // kleiner Helfer: gibt einen Stream zurück und setzt "any=true"
    auto L = [&]() -> std::ostream& { any = true; return log; };

    auto onoff = [](GLboolean b){ return b ? "enabled" : "disabled"; };

    // sicherer Größen-Logger
    auto logSizeChange = [&](const char* label, GLint64 a, GLint64 b){
        if (a != b) {
            L() << label << " size changed: " << a << " -> " << b << " bytes\n";
        }
        };

    // --- Kern-Bindings ------------------------------------------------------
    if (before.m_currentProgram != after.m_currentProgram)
        L() << "GL_CURRENT_PROGRAM changed: "
        << before.m_currentProgram << " -> " << after.m_currentProgram << "\n";

    if (before.m_vertexArrayBinding != after.m_vertexArrayBinding)
        L() << "GL_VERTEX_ARRAY_BINDING changed: "
        << before.m_vertexArrayBinding << " -> " << after.m_vertexArrayBinding << "\n";

    if (before.m_arrayBufferBinding != after.m_arrayBufferBinding)
        L() << "GL_ARRAY_BUFFER_BINDING changed: "
        << before.m_arrayBufferBinding << " -> " << after.m_arrayBufferBinding << "\n";

    if (before.m_elementArrayBufferBinding != after.m_elementArrayBufferBinding)
        L() << "GL_ELEMENT_ARRAY_BUFFER_BINDING changed: "
        << before.m_elementArrayBufferBinding << " -> " << after.m_elementArrayBufferBinding << "\n";

    // --- Buffer-Größen (ruhig & korrekt) -----------------------------------
    if (before.m_arrayBufferBinding != 0 &&
        before.m_arrayBufferBinding == after.m_arrayBufferBinding)
    {
        logSizeChange("GL_ARRAY_BUFFER", before.m_arrayBufferSize, after.m_arrayBufferSize);
    }

    if (before.m_vertexArrayBinding != 0 &&
        before.m_vertexArrayBinding == after.m_vertexArrayBinding &&
        before.m_elementArrayBufferBinding != 0 &&
        before.m_elementArrayBufferBinding == after.m_elementArrayBufferBinding)
    {
        logSizeChange("GL_ELEMENT_ARRAY_BUFFER",
            before.m_elementArrayBufferSize, after.m_elementArrayBufferSize);
    }

    // --- Vertex-Attribute ---------------------------------------------------
    if (before.m_vertexAttribInfos.size() != after.m_vertexAttribInfos.size()) {
        L() << "VertexAttrib count changed: "
            << before.m_vertexAttribInfos.size() << " -> "
            << after.m_vertexAttribInfos.size() << "\n";
    }

    auto byIndex = [](const GLVertexAttribInfo& a, const GLVertexAttribInfo& b){
        return a.index < b.index;
        };
    std::vector<GLVertexAttribInfo> vb = before.m_vertexAttribInfos;
    std::vector<GLVertexAttribInfo> va = after.m_vertexAttribInfos;
    std::sort(vb.begin(), vb.end(), byIndex);
    std::sort(va.begin(), va.end(), byIndex);

    size_t ib = 0, ia = 0;
    while (ib < vb.size() || ia < va.size()) {
        if (ib < vb.size() && ia < va.size() && vb[ib].index == va[ia].index) {
            const auto& b = vb[ib];
            const auto& a = va[ia];
            if (b.enabled      != a.enabled      ||
                b.size         != a.size         ||
                b.type         != a.type         ||
                b.normalized   != a.normalized   ||
                b.stride       != a.stride       ||
                b.bufferBinding!= a.bufferBinding||
                b.pointer      != a.pointer      ||
                b.divisor      != a.divisor)
            {
                L() << "VertexAttrib[" << b.index << "] changed:\n"
                    << "  Before: enabled=" << (b.enabled ? "YES":"NO")
                    << ", size=" << b.size
                    << ", type=" << b.type
                    << ", normalized=" << (b.normalized ? "YES":"NO")
                    << ", stride=" << b.stride
                    << ", buffer=" << b.bufferBinding
                    << ", pointer=" << b.pointer
                    << ", divisor=" << b.divisor << "\n"
                    << "  After : enabled=" << (a.enabled ? "YES":"NO")
                    << ", size=" << a.size
                    << ", type=" << a.type
                    << ", normalized=" << (a.normalized ? "YES":"NO")
                    << ", stride=" << a.stride
                    << ", buffer=" << a.bufferBinding
                    << ", pointer=" << a.pointer
                    << ", divisor=" << a.divisor << "\n";
            }
            ++ib; ++ia;
        } else if (ia < va.size() && (ib == vb.size() || va[ia].index < vb[ib].index)) {
            const auto& a = va[ia++];
            L() << "VertexAttrib[" << a.index << "] ADDED: "
                << "enabled=" << (a.enabled ? "YES":"NO")
                << ", size=" << a.size
                << ", type=" << a.type
                << ", normalized=" << (a.normalized ? "YES":"NO")
                << ", stride=" << a.stride
                << ", buffer=" << a.bufferBinding
                << ", pointer=" << a.pointer
                << ", divisor=" << a.divisor << "\n";
        } else {
            const auto& b = vb[ib++];
            L() << "VertexAttrib[" << b.index << "] REMOVED: "
                << "enabled=" << (b.enabled ? "YES":"NO")
                << ", size=" << b.size
                << ", type=" << b.type
                << ", normalized=" << (b.normalized ? "YES":"NO")
                << ", stride=" << b.stride
                << ", buffer=" << b.bufferBinding
                << ", pointer=" << b.pointer
                << ", divisor=" << b.divisor << "\n";
        }
    }

    // --- Enables ------------------------------------------------------------
    if (before.m_cullFaceEnabled != after.m_cullFaceEnabled)
        L() << "GL_CULL_FACE " << onoff(before.m_cullFaceEnabled)
        << " -> " << onoff(after.m_cullFaceEnabled) << "\n";
    if (before.m_depthTestEnabled != after.m_depthTestEnabled)
        L() << "GL_DEPTH_TEST " << onoff(before.m_depthTestEnabled)
        << " -> " << onoff(after.m_depthTestEnabled) << "\n";
    if (before.m_blendEnabled != after.m_blendEnabled)
        L() << "GL_BLEND " << onoff(before.m_blendEnabled)
        << " -> " << onoff(after.m_blendEnabled) << "\n";
    if (before.m_polyOffsetFillEnabled != after.m_polyOffsetFillEnabled)
        L() << "GL_POLYGON_OFFSET_FILL " << onoff(before.m_polyOffsetFillEnabled)
        << " -> " << onoff(after.m_polyOffsetFillEnabled) << "\n";
    if (before.m_polyOffsetLineEnabled != after.m_polyOffsetLineEnabled)
        L() << "GL_POLYGON_OFFSET_LINE " << onoff(before.m_polyOffsetLineEnabled)
        << " -> " << onoff(after.m_polyOffsetLineEnabled) << "\n";

    // --- Rasterizer/Depth/Blend/Viewport/Scissor/Clear ---------------------
    if (before.m_polygonMode[0] != after.m_polygonMode[0] ||
        before.m_polygonMode[1] != after.m_polygonMode[1])
        L() << "GL_POLYGON_MODE changed: ("
        << before.m_polygonMode[0] << ", " << before.m_polygonMode[1]
        << ") -> ("
        << after.m_polygonMode[0]  << ", " << after.m_polygonMode[1]  << ")\n";

    if (before.m_cullFaceMode != after.m_cullFaceMode)
        L() << "GL_CULL_FACE_MODE changed: "
        << before.m_cullFaceMode << " -> " << after.m_cullFaceMode << "\n";

    if (before.m_frontFace != after.m_frontFace)
        L() << "GL_FRONT_FACE changed: "
        << before.m_frontFace << " -> " << after.m_frontFace << "\n";

    if (before.m_depthFunc != after.m_depthFunc)
        L() << "GL_DEPTH_FUNC changed: "
        << before.m_depthFunc << " -> " << after.m_depthFunc << "\n";

    if (before.m_depthMask != after.m_depthMask)
        L() << "GL_DEPTH_WRITEMASK changed: "
        << (before.m_depthMask ? "true" : "false") << " -> "
        << (after.m_depthMask  ? "true" : "false") << "\n";

    if (before.m_lineWidth != after.m_lineWidth)
        L() << "GL_LINE_WIDTH changed: "
        << before.m_lineWidth << " -> " << after.m_lineWidth << "\n";

    if (before.m_pointSize != after.m_pointSize)
        L() << "GL_POINT_SIZE changed: "
        << before.m_pointSize << " -> " << after.m_pointSize << "\n";

    if (before.m_polygonOffsetFactor != after.m_polygonOffsetFactor ||
        before.m_polygonOffsetUnits  != after.m_polygonOffsetUnits)
        L() << "GL_POLYGON_OFFSET changed: factor "
        << before.m_polygonOffsetFactor << " -> " << after.m_polygonOffsetFactor
        << ", units " << before.m_polygonOffsetUnits << " -> " << after.m_polygonOffsetUnits << "\n";

    if (before.m_depthRange[0] != after.m_depthRange[0] ||
        before.m_depthRange[1] != after.m_depthRange[1])
        L() << "GL_DEPTH_RANGE changed: ("
        << before.m_depthRange[0] << ", " << before.m_depthRange[1] << ") -> ("
        << after.m_depthRange[0]  << ", " << after.m_depthRange[1]  << ")\n";

    auto any4 = [](const GLfloat a[4], const GLfloat b[4]){
        return a[0]!=b[0] || a[1]!=b[1] || a[2]!=b[2] || a[3]!=b[3];
        };
    if (any4(before.m_blendColor, after.m_blendColor))
        L() << "GL_BLEND_COLOR changed\n";
    if (before.m_blendSrcRGB != after.m_blendSrcRGB)
        L() << "GL_BLEND_SRC_RGB changed: "
        << before.m_blendSrcRGB << " -> " << after.m_blendSrcRGB << "\n";
    if (before.m_blendDstRGB != after.m_blendDstRGB)
        L() << "GL_BLEND_DST_RGB changed: "
        << before.m_blendDstRGB << " -> " << after.m_blendDstRGB << "\n";
    if (before.m_blendSrcAlpha != after.m_blendSrcAlpha)
        L() << "GL_BLEND_SRC_ALPHA changed: "
        << before.m_blendSrcAlpha << " -> " << after.m_blendSrcAlpha << "\n";
    if (before.m_blendDstAlpha != after.m_blendDstAlpha)
        L() << "GL_BLEND_DST_ALPHA changed: "
        << before.m_blendDstAlpha << " -> " << after.m_blendDstAlpha << "\n";
    if (before.m_blendEquationRGB != after.m_blendEquationRGB)
        L() << "GL_BLEND_EQUATION_RGB changed: "
        << before.m_blendEquationRGB << " -> " << after.m_blendEquationRGB << "\n";
    if (before.m_blendEquationAlpha != after.m_blendEquationAlpha)
        L() << "GL_BLEND_EQUATION_ALPHA changed: "
        << before.m_blendEquationAlpha << " -> " << after.m_blendEquationAlpha << "\n";

    if (before.m_viewport[0] != after.m_viewport[0] ||
        before.m_viewport[1] != after.m_viewport[1] ||
        before.m_viewport[2] != after.m_viewport[2] ||
        before.m_viewport[3] != after.m_viewport[3])
        L() << "GL_VIEWPORT changed: ("
        << before.m_viewport[0] << ", " << before.m_viewport[1] << ", "
        << before.m_viewport[2] << ", " << before.m_viewport[3] << ") -> ("
        << after.m_viewport[0]  << ", " << after.m_viewport[1]  << ", "
        << after.m_viewport[2]  << ", " << after.m_viewport[3]  << ")\n";

    if (before.m_scissorBox[0] != after.m_scissorBox[0] ||
        before.m_scissorBox[1] != after.m_scissorBox[1] ||
        before.m_scissorBox[2] != after.m_scissorBox[2] ||
        before.m_scissorBox[3] != after.m_scissorBox[3])
        L() << "GL_SCISSOR_BOX changed: ("
        << before.m_scissorBox[0] << ", " << before.m_scissorBox[1] << ", "
        << before.m_scissorBox[2] << ", " << before.m_scissorBox[3] << ") -> ("
        << after.m_scissorBox[0]  << ", " << after.m_scissorBox[1]  << ", "
        << after.m_scissorBox[2]  << ", " << after.m_scissorBox[3]  << ")\n";

    if (any4(before.m_clearColor, after.m_clearColor))
        L() << "GL_COLOR_CLEAR_VALUE changed\n";
    if (before.m_clearDepth != after.m_clearDepth)
        L() << "GL_DEPTH_CLEAR_VALUE changed: "
        << before.m_clearDepth << " -> " << after.m_clearDepth << "\n";

    if (before.m_activeTexture != after.m_activeTexture)
        L() << "GL_ACTIVE_TEXTURE changed: "
        << before.m_activeTexture << " -> " << after.m_activeTexture << "\n";
    if (before.m_textureBinding2D != after.m_textureBinding2D)
        L() << "GL_TEXTURE_BINDING_2D changed: "
        << before.m_textureBinding2D << " -> " << after.m_textureBinding2D << "\n";
    if (before.m_framebufferBinding != after.m_framebufferBinding)
        L() << "GL_FRAMEBUFFER_BINDING changed: "
        << before.m_framebufferBinding << " -> " << after.m_framebufferBinding << "\n";
    if (before.m_renderbufferBinding != after.m_renderbufferBinding)
        L() << "GL_RENDERBUFFER_BINDING changed: "
        << before.m_renderbufferBinding << " -> " << after.m_renderbufferBinding << "\n";

    // --- Ausgabe nur bei Änderungen -----------------------------------------
    if (any) {
        if (!info.empty()) std::cout << info << "\n";
        std::cout << log.str();
    }
}


void GLState::printAllExistingBuffers(GLuint maxID) {
    std::vector<GLenum> targets = { GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER };
    std::cout << "=== Overview of All Buffers ===" << std::endl;
    for (GLuint i = 1; i <= maxID; ++i) {
        if (glIsBuffer(i)) {
            GLState::printCandidateBuffer(i, targets);
        }
    }
}


void GLState::printCurrentContext() {
#ifdef _WIN32
    HGLRC context = wglGetCurrentContext();
    HDC device = wglGetCurrentDC();
    std::cout << "Current OpenGL Context (wglGetCurrentContext): " << context << std::endl;
    std::cout << "Current Device Context (wglGetCurrentDC): " << device << std::endl;
#elif defined(__linux__)
    GLXContext context = glXGetCurrentContext();
    Display* display = glXGetCurrentDisplay();
    std::cout << "Current OpenGL Context (glXGetCurrentContext): " << context << std::endl;
    std::cout << "Current Display (glXGetCurrentDisplay): " << display << std::endl;
#else
    std::cout << "Context query not implemented for this system." << std::endl;
#endif
}


void GLState::printBufferContents(GLenum target, GLuint buffer, size_t sizeInBytes) {
    GLState::printCurrentContext();
    GLint oldBinding = 0;
    if (target == GL_ARRAY_BUFFER) {
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBinding);
    }
    else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &oldBinding);
    }

    std::cout << "Target: " << ((target == GL_ARRAY_BUFFER) ? "GL_ARRAY_BUFFER" : "GL_ELEMENT_ARRAY_BUFFER") << std::endl;
    std::cout << "Buffer ID: " << buffer << std::endl;
    std::cout << "Previously bound buffer: " << oldBinding
        << ((static_cast<GLuint>(oldBinding) == buffer) ? " (same)" : " (different)") << std::endl;

    glBindBuffer(target, buffer);

    void* ptr = glMapBuffer(target, GL_READ_ONLY);
    if (ptr == nullptr) {
        std::cerr << "glMapBuffer() failed." << std::endl;
        glBindBuffer(target, oldBinding);
        return;
    }

    if (target == GL_ARRAY_BUFFER) {
        float* data = static_cast<float*>(ptr);
        size_t numElements = sizeInBytes / sizeof(float);
        std::cout << "Array Buffer Contents (" << numElements << " float elements):" << std::endl;
        for (size_t i = 0; i < numElements; i++) {
            std::cout << "Element " << i << ": " << data[i] << std::endl;
        }
    }
    else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        unsigned short* data = static_cast<unsigned short*>(ptr);
        size_t numElements = sizeInBytes / sizeof(unsigned short);
        std::cout << "Element Array Buffer Contents (" << numElements << " unsigned short elements):" << std::endl;
        for (size_t i = 0; i < numElements; i++) {
            std::cout << "Element " << i << ": " << data[i] << std::endl;
        }
    }
    else {
        std::cerr << "Unknown buffer target." << std::endl;
    }

    glUnmapBuffer(target);
    glBindBuffer(target, oldBinding);
}


void GLState::printVAOAttributes(GLuint vao) {
    GLint previousVAO = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVAO);
    printCurrentContext();
    glBindVertexArray(vao);
    GLint maxAttribs = 0;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxAttribs);
    std::cout << "Maximum Vertex Attributes: " << maxAttribs << std::endl;

    for (GLint i = 0; i < maxAttribs; ++i) {
        GLint enabled = 0;
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
        std::cout << "Attribute " << i << " enabled: " << enabled << std::endl;

        if (enabled) {
            GLint size = 0, type = 0, stride = 0, bufferBinding = 0;
            glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size);
            glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type);
            glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &stride);
            glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &bufferBinding);
            std::cout << "  Size: " << size
                << ", Type: " << type
                << ", Stride: " << stride
                << ", Buffer Binding: " << bufferBinding << std::endl;
        }
    }
    glBindVertexArray(previousVAO);
}


void GLState::printCandidateBuffer(GLuint candidate, const std::vector<GLenum>& targets) {
    std::cout << "Buffer Handle: " << candidate << std::endl;
    for (auto target : targets) {
        std::string targetName;
        switch (target) {
        case GL_ARRAY_BUFFER: targetName = "GL_ARRAY_BUFFER"; break;
        case GL_ELEMENT_ARRAY_BUFFER: targetName = "GL_ELEMENT_ARRAY_BUFFER"; break;
        default: targetName = "Unknown Target"; break;
        }
        GLint oldBinding = 0;
        if (target == GL_ARRAY_BUFFER)
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBinding);
        else if (target == GL_ELEMENT_ARRAY_BUFFER)
            glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &oldBinding);
        bool isBound = (static_cast<GLuint>(oldBinding) == candidate);
        std::cout << "  Target: " << targetName
            << ", Currently Bound: " << (isBound ? "YES" : "NO") << std::endl;

        glBindBuffer(target, candidate);
        GLint bufferSize = 0;
        glGetBufferParameteriv(target, GL_BUFFER_SIZE, &bufferSize);
        std::cout << "    Buffer Type: " << targetName << ", Size: " << bufferSize << " bytes" << std::endl;
        if (bufferSize > 0) {
            void* ptr = glMapBuffer(target, GL_READ_ONLY);
            if (ptr) {
                if (target == GL_ARRAY_BUFFER) {
                    float* data = static_cast<float*>(ptr);
                    int numElements = bufferSize / sizeof(float);
                    std::cout << "    Elements (first 4): ";
                    for (int i = 0; i < std::min(numElements, 4); ++i)
                        std::cout << data[i] << " ";
                    std::cout << "\n    Elements (last 4): ";
                    for (int i = std::max(0, numElements - 4); i < numElements; ++i)
                        std::cout << data[i] << " ";
                    std::cout << std::endl;
                }
                else if (target == GL_ELEMENT_ARRAY_BUFFER) {
                    unsigned short* data = static_cast<unsigned short*>(ptr);
                    int numElements = bufferSize / sizeof(unsigned short);
                    std::cout << "    Elements (first 4): ";
                    for (int i = 0; i < std::min(numElements, 4); ++i)
                        std::cout << data[i] << " ";
                    std::cout << "\n    Elements (last 4): ";
                    for (int i = std::max(0, numElements - 4); i < numElements; ++i)
                        std::cout << data[i] << " ";
                    std::cout << std::endl;
                }
                glUnmapBuffer(target);
            }
            else {
                std::cout << "    Buffer mapping failed." << std::endl;
            }
        }
        else {
            std::cout << "    Buffer size is 0." << std::endl;
        }
        glBindBuffer(target, oldBinding);
    }
    std::cout << std::endl;
}
