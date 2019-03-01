#
# Copyright 2017-2019 Stanislav Pidhorskyi. All rights reserved.
# License: https://raw.githubusercontent.com/podgorskiy/impy/master/LICENSE.txt
#

from setuptools import setup, Extension, find_packages
from distutils.errors import *
from distutils.dep_util import newer_group
from distutils import log
from distutils.command.build_ext import build_ext

from codecs import open
import os
import sys
import platform
import re

target_os = 'none'

if sys.platform == 'darwin':
    target_os = 'darwin'
elif os.name == 'posix':
    target_os = 'posix'
elif platform.system() == 'Windows':
    target_os = 'win32'
    
here = os.path.abspath(os.path.dirname(__file__))

#with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
#    long_description = f.read()
long_description = ""

def filter_sources(sources):
    """Filters sources into c, cpp and objc"""
    cpp_ext_match = re.compile(r'.*[.](cpp|cxx|cc)\Z', re.I).match
    c_ext_match = re.compile(r'.*[.](c|C)\Z', re.I).match
    objc_ext_match = re.compile(r'.*[.]m\Z', re.I).match

    c_sources = []
    cpp_sources = []
    objc_sources = []
    other_sources = []
    for source in sources:
        if c_ext_match(source):
            c_sources.append(source)
        elif cpp_ext_match(source):
            cpp_sources.append(source)
        elif objc_ext_match(source):
            objc_sources.append(source)
        else:
            other_sources.append(source)
    return c_sources, cpp_sources, objc_sources, other_sources


def build_extension(self, ext):
    """Modified version of build_extension method from distutils.
       Can handle compiler args for different files"""

    sources = ext.sources
    if sources is None or not isinstance(sources, (list, tuple)):
        raise DistutilsSetupError(
              "in 'ext_modules' option (extension '%s'), "
              "'sources' must be present and must be "
              "a list of source filenames" % ext.name)

    sources = list(sources)
    ext_path = self.get_ext_fullpath(ext.name)
    depends = sources + ext.depends
    if not (self.force or newer_group(depends, ext_path, 'newer')):
        log.debug("skipping '%s' extension (up-to-date)", ext.name)
        return
    else:
        log.info("building '%s' extension", ext.name)

    sources = self.swig_sources(sources, ext)

    extra_args = ext.extra_compile_args or []
    extra_c_args = getattr(ext, "extra_compile_c_args", [])
    extra_cpp_args = getattr(ext, "extra_compile_cpp_args", [])
    extra_objc_args = getattr(ext, "extra_compile_objc_args", [])
    macros = ext.define_macros[:]
    for undef in ext.undef_macros:
        macros.append((undef,))

    c_sources, cpp_sources, objc_sources, other_sources = filter_sources(sources)

    def _compile(src, args):
        return self.compiler.compile(src,
                                     output_dir=self.build_temp,
                                     macros=macros,
                                     include_dirs=ext.include_dirs,
                                     debug=self.debug,
                                     extra_postargs=extra_args + args,
                                     depends=ext.depends)

    objects = []
    objects += _compile(c_sources, extra_c_args)
    objects += _compile(cpp_sources, extra_cpp_args)
    objects += _compile(objc_sources, extra_objc_args)
    objects += _compile(other_sources, [])

    self._built_objects = objects[:]
    if ext.extra_objects:
        objects.extend(ext.extra_objects)

    extra_args = ext.extra_link_args or []

    language = ext.language or self.compiler.detect_language(sources)
    self.compiler.link_shared_object(
        objects, ext_path,
        libraries=self.get_libraries(ext),
        library_dirs=ext.library_dirs,
        runtime_library_dirs=ext.runtime_library_dirs,
        extra_postargs=extra_args,
        export_symbols=self.get_export_symbols(ext),
        debug=self.debug,
        build_temp=self.build_temp,
        target_lang=language)

# patching
build_ext.build_extension = build_extension

definitions = {
    'darwin': [],
    'posix': [],
    'win32': [("_CRT_SECURE_NO_WARNINGS", 1), ("NOMINMAX", 1)],
}

extra_compile_args = {
    'darwin': [],
    'posix': ['-O3', '-funroll-loops', '-march=native', '-mfpmath=sse'],
    'win32': ['/MT', '/GL', '/GR-'],
}

extra_compile_cpp_args = {
    'darwin': ['-std=c++11'],
    'posix': ['-std=c++11'],
    'win32': [],
}

extension = Extension("_hashranking",
                             ['hashranking.cpp'],
                             define_macros = definitions[target_os],
                             include_dirs=["pybind11/include"],
                             extra_compile_args=extra_compile_args[target_os],
                             extra_link_args=[],
                             libraries = [])

extension.extra_compile_cpp_args = extra_compile_cpp_args[target_os]

setup(
    name='hashranking',

    version='0.0.1',

    description='fast procedures for forking with hashes',
    long_description=long_description,

    #url='https://github.com/podgorskiy/bimpy',

    author='Stanislav Pidhorskyi',
    author_email='stanislav@podgorskiy.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    packages=['hashranking'],

    ext_modules=[extension],
)
