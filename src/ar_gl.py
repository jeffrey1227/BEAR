import ctypes
import os

from pyglet.gl import *
from pywavefront import visualization, Wavefront

window = pyglet.window.Window(width=1080, height=720, resizable=True)

root_path = os.path.dirname(__file__)

bball = Wavefront(os.path.join(root_path, 'data/basketball-lowpoly/basketBall_OBJ.obj'), collect_faces=True)

rotation = 0.0
lightfv = ctypes.c_float * 4

@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))

    draw_model(bball, -2.0, 0.0, -10.0)
    draw_model(bball, 0.0, 0.0, -10.0)
    draw_model(bball, 1.0, 0.0, -10.0)
    draw_model(bball, 2.0, 0.0, -10.0)
    draw_model(bball, 3.0, 0.0, -10.0)
    draw_model(bball, 4.0, 0.0, -10.0)
    draw_model(bball, 6.0, 0.0, -10.0)


def draw_model(model, x, y, z):
    glLoadIdentity()
    glTranslated(x, y, z)

    visualization.draw(model)


def update(dt):
    global rotation
    rotation += 90.0 * dt

    if rotation > 720.0:
        rotation = 0.0


pyglet.clock.schedule(update)
pyglet.app.run()