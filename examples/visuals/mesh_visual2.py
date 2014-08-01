# -*- coding: utf-8 -*-
# Copyright (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" A Mesh Visual that uses the new shader Function.
"""

from __future__ import division

import numpy as np

#from .visual import Visual
#from ..shader.function import Function, Variable
#from ...import gloo

from vispy.scene.visuals.visual import Visual
from vispy.scene.shaders import ModularProgram, Function, Variable, Varying
from vispy import gloo


## Snippet templates (defined as string to force user to create fresh Function)
# Consider these stored in a central location in vispy ...


vertex_template = """
void main() {
}
"""

fragment_template = """
void main() {
}
"""

phong_template = """
vec4 phong_shading(vec4 color) {
    vec3 norm = normalize($normal.xyz);
    vec3 light = normalize($light_dir.xyz);
    float p = dot(light, norm);
    p = (p < 0. ? 0. : p);
    vec4 diffuse = $light_color * p;
    diffuse.a = 1.0;
    p = dot(reflect(light, norm), vec3(0,0,1));
    if (p < 0.0) {
        p = 0.0;
    }
    vec4 specular = $light_color * 5.0 * pow(p, 100.);
    return color * ($ambient + diffuse) + specular;
}
"""

## Functions that can be used as is (don't have template variables)
# Consider these stored in a central location in vispy ...

color3to4 = Function("""
vec4 color3to4(vec3 rgb) {
    return vec4(rgb, 1.0);
}
""")

stub4 = Function("vec4 stub4(vec4 value) { return value; }")
stub3 = Function("vec4 stub3(vec3 value) { return value; }")


## Actual code

class SuperVariable4(object):
    """ Provides an easy way to bring vec4 data into a shader
    
    This class takes care of carying attribute data from vertex to
    fragment shader via a varying, and the conversion of float/vec2/vec3
    data to vec4 data.
    """
    
    _code1 = 'vec4 val1to4(float val) { return vec4(val, 0.0, 0.0, 1.0);}'
    _code2 = 'vec4 val2to4(vec2 val) { return vec4(val, 0.0, 1.0);}'
    _code3 = 'vec4 val3to4(vec3 val) { return vec4(val, 1.0);}'
    _code4 = 'vec4 stub4(vec4 val) { return val;}'
    
    def __init__(self, name, value=None, n=4):
        self._variable = Variable(name) 
        self._varying = Varying('a_' + name)
        self._varying.link(self._variable)
        
        # Define proxies
        self._proxies = {}
        self._proxies['float'] = Function(self._code1)
        self._proxies['vec2'] = Function(self._code2)
        self._proxies['vec3'] = Function(self._code3)
        self._proxies['vec4'] = Function(self._code4)
        # Turn into FunctionCall objects
        for key, val in self._proxies.items():
            if val.name.startswith('stub'):
                val = lambda x:x
            self._proxies[key] = val(self._variable), val(self._varying)
        
        if value is not None:
            self.value = value
    
    @property
    def value(self):
        return self._variable.value
    
    @value.setter
    def value(self, value):
        self._variable.value = value
    
    
    def apply(self, fun1, fun2=None):
        """ Apply this variable. If one function object is given, simply
        applies the value to that. If two are given, they are considered
        vertex and fragment shader, and a varying is used to communicate the
        value between the two.
        """
        name, dtype = self._variable.name, self._variable.dtype
        if fun2 is None:
            fun1[name] = self._proxies[dtype][0]
        else:
            fun2[name] = self._proxies[dtype][1]
            fun1[self._varying] = self._proxies[dtype][0]


class Mesh(Visual):
    
    def __init__(self, parent, 
                 vertices, faces=None, normals=None, values=None):
        Visual.__init__(self, parent)
        
        # Create a program
        self._program = ModularProgram(vertex_template, fragment_template)
        
        # Define how we are going to specify position and color
        self._program.vert['gl_Position'] = 'vec4($position, 1.0)'
        self._program.frag['gl_FragColor'] = '$light($color)'
        
        # Define variable related to color
        self._colorvar = SuperVariable4('color')
        
        # Init
        self.shading = 'plain'
        #
        self.set_vertices(vertices)
        self.set_faces(faces)
        self.set_normals(normals)
        self.set_values(values)
    
    def set_vertices(self, vertices):
        vertices = gloo.VertexBuffer(vertices)
        self._program.vert['position'] = vertices
    
    def set_faces(self, faces):
        if faces is not None:
            self._faces = gloo.IndexBuffer(faces)
        else:
            self._faces = None
    
    def set_normals(self, normals):
        self._normals = normals
        self.shading = self.shading  # Update
    
    def set_values(self, values):
        
        # todo: reuse vertex buffer is possible (use set_data)
        # todo: we may want to clear any color vertex buffers
        
        if isinstance(values, tuple):
            if len(values) not in (3, 4):
                raise ValueError('Color tuple must have 3 or 4 values.')
            # Single value (via a uniform)
            self._colorvar.value = [float(v) for v in values]
            self._colorvar.apply(self._program.frag)
            
        elif isinstance(values, np.ndarray):
            # A value per vertex, via a VBO
            
            if values.shape[1] == 1:
                # Look color up in a colormap
                raise NotImplementedError()
            
            elif values.shape[1] == 2:
                # Look color up in a texture
                raise NotImplementedError()
            
            if values.shape[1] in (3, 4):
                # Explicitly set color per vertex
                if isinstance(self._colorvar.value, gloo.VertexBuffer):
                    # todo: set_data should check whether this is allowed
                    self._colorvar.value.set_data(values)
                else:
                    self._colorvar.value = gloo.VertexBuffer(values)
                self._colorvar.apply(self._program.vert, self._program.frag)
            else:
                raise ValueError('Mesh values must be NxM, with M 1,2,3 or 4.')
        else:
            raise ValueError('Mesh values must be NxM array or color tuple')
        
        print(self._program._need_build)
    
    @property
    def shading(self):
        """ The shading method used.
        """
        return self._shading
    
    @shading.setter
    def shading(self, value):
        assert value in ('plain', 'flat', 'phong')
        self._shading = value
        # todo: add gouroud shading
        # todo: allow flat shading even if vertices+faces is specified.
        if value == 'plain':
            self._program.frag['light'] = stub4
        
        elif value == 'flat':
            pass
            
        elif value == 'phong':
            assert self._normals is not None
            # Apply phong function, 
            phong = Function(phong_template)
            self._program.frag['light'] = phong
            # Normal data comes via vertex shader
            phong['normal'] = Varying('v_normal')
            var = gloo.VertexBuffer(self._normals)
            self._program.vert[phong['normal']] = var
            # Additional phong proprties
            phong['light_dir'] = 'vec3(1.0, 1.0, 1.0)'
            phong['light_color'] = 'vec4(1.0, 1.0, 1.0, 1.0)'
            phong['ambient'] = 'vec4(0.3, 0.3, 0.3, 1.0)'
            # todo: light properties should be queried from the SubScene
            # instance.
    
    def draw(self, event):
        # Draw
        self._program.draw('triangles', self._faces)


## The code to show it ...

if __name__ == '__main__':
    
    from vispy import app
    from vispy.util.meshdata import sphere
    
    mdata = sphere(20, 20)
    faces = mdata.faces()
    verts = mdata.vertices() / 4.0
    verts_flat = mdata.vertices(indexed='faces').reshape(-1, 3) / 4.0
    normals = mdata.vertex_normals()
    normals_flat = mdata.vertices(indexed='faces').reshape(-1, 3) / 4.0
    colors = np.random.uniform(0.2, 0.8, (verts.shape[0], 3)).astype('float32')
    colors_flat = np.random.uniform(0.2, 0.8, 
                                    (verts_flat.shape[0], 3)).astype('float32')
    
    class Canvas(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, close_keys='escape')
            self.size = 700,    700
            self.meshes = []
            
            # A plain mesh with uniform color
            offset = np.array([-0.7, 0.7, 0.0], 'float32')
            mesh = Mesh(None, verts+offset, faces, None,
                        values=(1.0, 0.4, 0.0, 1.0))
            self.meshes.append(mesh)
            
            # A colored mesh with one color per phase
            offset = np.array([0.0, 0.7, 0.0], 'float32')
            mesh = Mesh(None, verts_flat+offset, None, None,
                        values=colors_flat)
            self.meshes.append(mesh)
            
            # Same mesh but using faces, so we get interpolation of color
            offset = np.array([0.7, 0.7, 0.0], 'float32')
            mesh = Mesh(None, verts+offset, faces, None, values=colors)
            self.meshes.append(mesh)
            
            # Flat phong shading
            offset = np.array([0.0, 0.0, 0.0], 'float32')
            mesh = Mesh(None, verts_flat+offset, None, normals_flat, 
                        values=colors_flat)
            mesh.shading = 'phong'
            self.meshes.append(mesh)
            
            # Full phong shading
            offset = np.array([0.7, 0.0, 0.0], 'float32')
            mesh = Mesh(None, verts+offset, faces, normals, values=colors)
            mesh.shading = 'phong'
            self.meshes.append(mesh)
        
        def on_draw(self, event):
            gloo.clear()
            gloo.set_viewport(0, 0, *self.size)
            for mesh in self.meshes:
                mesh.draw(self)
    
    c = Canvas()
    m = c.meshes[0]
    c.show()
    app.run()
