#
# Translation functions for vr-prepare
# Visenso GmbH
# (c) 2012
#
# $Id:$

### Replaces the qt loalization methods with 
### the ones used by gettext.
###
### import and wrap compiler.UICompiler and

from PyQt5.uic.Compiler import compiler, qtproxies, indenter

# pylint: disable=C0103
class _UICompiler(compiler.UICompiler):
    """Speciallized compiler for qt .ui files."""

    def createToplevelWidget(self, classname, widgetname):
        o = indenter.getIndenter()
        o.level = 0
        o.write('from vtrans import coTranslate')
        o.write(' ')
        return super(_UICompiler, self).createToplevelWidget(classname, widgetname)

compiler.UICompiler = _UICompiler

### wrap qtproxies.i18n_string
class _i18n_string(qtproxies.i18n_string):
    """Provide a translated text."""

    def __str__(self):
        return "coTranslate(\"\"\"%s\"\"\")" % self.string.replace("\"", "\\\"").replace("\\", "\\\\")

qtproxies.i18n_string = _i18n_string

### run /usr/bin/pyuic4
# there's no main function, so just import the module
import PyQt5.uic.pyuic
