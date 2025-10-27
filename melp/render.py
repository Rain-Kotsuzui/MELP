import warp as wp
import warp.render as wr
from .particleSystem import *
from typing import Any


class render:
    renderer: wr.OpenGLRenderer

    def __init__(self, title='',
                 scaling=1.0,
                 fps=60,
                 up_axis='Y',
                 screen_width=1024,
                 screen_height=768,
                 near_plane=0.1,
                 far_plane=100.0,
                 camera_fov=45.0,
                 camera_pos=(0.0, 2.0, 10.0),
                 camera_front=(0.0, 0.0, -1.0),
                 camera_up=(0.0, 1.0, 0.0),
                 background_color=(0.53, 0.8, 0.92),
                 draw_grid=True,
                 draw_sky=True,
                 draw_axis=True,
                 show_info=True,
                 render_wireframe=False,
                 render_depth=False,
                 axis_scale=1.0,
                 vsync=False,
                 headless=None,
                 enable_backface_culling=True,
                 enable_mouse_interaction=True,
                 enable_keyboard_interaction=True,
                 device=None,
                 use_legacy_opengl=None) -> None:
        self.renderer = wr.OpenGLRenderer(title,
                                          scaling,
                                          fps,
                                          up_axis,
                                          screen_width,
                                          screen_height,
                                          near_plane,
                                          far_plane,
                                          camera_fov,
                                          camera_pos,
                                          camera_front,
                                          camera_up,
                                          background_color,
                                          draw_grid,
                                          draw_sky,
                                          draw_axis,
                                          show_info,
                                          render_wireframe,
                                          render_depth,
                                          axis_scale,
                                          vsync,
                                          headless,
                                          enable_backface_culling,
                                          enable_mouse_interaction,
                                          enable_keyboard_interaction,
                                          device,
                                          use_legacy_opengl)

    def rend(self, Bubble: Any) -> None:
        N = len(Bubble)
        self.renderer.begin_frame()
        for i in range(N):
            self.renderer.render_points(
                name="Eposs",
                points=Bubble[i].Eposs,
                colors=(0.0, 1.0, 0.0),
                radius=0.03
            )
            self.renderer.render_points(
                name="Lposs",
                points=Bubble[i].Lposs,
                colors=(1.0, 0.0, 0.0),
                radius=0.03
            )
            self.renderer.render_points(
                name="Enorm",
                points=Bubble[i].Enorm,
                colors=(0.0, 1.0, 0.0),
                radius=0.01
            )
            self.renderer.render_points(
                name="Lnorm",
                points=Bubble[i].Lnorm,
                colors=(1.0, 0.0, 0.0),
                radius=0.01
            )
        self.renderer.render_points(
            name="center",
            points=Bubble[i].center,
            colors=(1.0, 0.0, 1.0),
            radius=0.1
        )
        self.renderer.end_frame()

    def running(self) -> bool:
        return self.renderer.is_running()
