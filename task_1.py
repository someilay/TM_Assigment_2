import numpy as np
import copy
from manim import *
from math import cos, sin, sqrt, acos
from scipy.optimize import fsolve
from typing import Union as TUnion
from helpers import get_mut_dot, c2p, get_line, create_mut_obj, get_vector, get_vec_info, get_dot_title, \
    get_vector_title, sign


class Task1(Scene):
    # Configs
    AXES_CONFIG = {
        'x_range': [-35, 115],
        'y_range': [-60, 60],
        'x_length': 7.5,
        'y_length': 6,
        'axis_config': {
            'stroke_width': 1,
            'tip_width': 0.15,
            'tip_height': 0.15,
            'tick_size': 0.01,
        },
        'x_axis_config': {
            'numbers_to_include': np.arange(-30, 115, 10),
            'numbers_with_elongated_ticks': np.arange(-30, 115, 10),
            'font_size': 16,
            'line_to_number_buff': MED_SMALL_BUFF / 3,
        },
        'y_axis_config': {
            'numbers_to_include': np.arange(-50, 60, 10),
            'numbers_with_elongated_ticks': np.arange(-50, 60, 10),
            'font_size': 16,
            'line_to_number_buff': MED_SMALL_BUFF / 3,
        },
    }
    VECTORS_KWARGS = {
        'stroke_width': 3,
        'buff': 0,
        'max_tip_length_to_length_ratio': 0.12,
        'max_stroke_width_to_length_ratio': 6,
    }
    MATRIX_CONFIG = {
        'stroke_width': 1,
        'element_alignment_corner': LEFT,
        'element_to_mobject_config': {
            'num_decimal_places': 3,
            'font_size': 28,
        }
    }

    PHI_0 = PI / 3
    ANG_VEL_AO1 = 2
    AO1 = 21
    BO2 = 25
    FO3 = 20
    AB = 54
    CD = 69
    AC = CD / 3
    CE = 35
    EF = 32
    A = 56
    B = 10
    C = 26
    D = 16
    E = 25

    FIXED_POINTS_COORDS = [
        np.array((0, 0, 0)),  # O1
        np.array((A, -C, 0)),  # O2
        np.array((A + B, D + E, 0))  # O3
    ]

    # Critical angles
    GAMMA = acos(A / sqrt(A ** 2 + C ** 2))
    MAX_THETA_1 = acos((AO1 ** 2 + A ** 2 + C ** 2 - (AB + BO2) ** 2) / (2 * AO1 * sqrt(A ** 2 + C ** 2))) - GAMMA
    MAX_THETA_2 = 2 * PI - (MAX_THETA_1 + 2 * GAMMA)

    POINT_RADIUS = DEFAULT_DOT_RADIUS / 2
    POINT_COLOR = WHITE

    C_ROT_MATRIX = np.array(
        [
            [247 / 828, -sqrt(1 - (247 / 828) ** 2), 0],
            [sqrt(1 - (247 / 828) ** 2), 247 / 828, 0],
            [0, 0, 0]
        ]
    )

    DT = 0.02
    # Critical moments of time when
    CRITICAL_TIME_1 = (MAX_THETA_1 - PHI_0) / ANG_VEL_AO1
    CRITICAL_TIME_2 = (MAX_THETA_2 - PHI_0) / ANG_VEL_AO1
    # Mechanism exist when t belongs TIME_INTERVALS[0] and IME_INTERVALS[1], TIME_INTERVALS[2] and IME_INTERVALS[3]
    TIME_INTERVALS = [
        0,
        CRITICAL_TIME_1 - 10 * DT,
        CRITICAL_TIME_2 + 10 * DT,
        2 * PI / ANG_VEL_AO1
    ]

    # Buffered coordinates for B and F
    BUFFERED_B_COORDS = dict()
    BUFFERED_F_COORDS = dict()
    POSS_B_COORD = np.array((60, 0))
    POSS_F_COORD = np.array((90, 40))

    # Laws of motion, coord == coordinates, vel == velocity, acc == acceleration, _t == tangent, _n == normal
    @staticmethod
    def a_coord(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.AO1 * cos(source.ANG_VEL_AO1 * t + source.PHI_0),
                source.AO1 * sin(source.ANG_VEL_AO1 * t + source.PHI_0),
                0
            )
        )

    @staticmethod
    def a_vel(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                -source.ANG_VEL_AO1 * source.AO1 * sin(source.ANG_VEL_AO1 * t + source.PHI_0),
                source.ANG_VEL_AO1 * source.AO1 * cos(source.ANG_VEL_AO1 * t + source.PHI_0),
                0
            )
        )

    @staticmethod
    def a_acc(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                -(source.ANG_VEL_AO1 ** 2) * source.AO1 * cos(source.ANG_VEL_AO1 * t + source.PHI_0),
                -(source.ANG_VEL_AO1 ** 2) * source.AO1 * sin(source.ANG_VEL_AO1 * t + source.PHI_0),
                0
            )
        )

    @staticmethod
    def a_acc_t(t: float) -> np.ndarray:
        source = Task1
        a_vel = source.a_vel(t)
        n_a_vel = np.linalg.norm(a_vel)
        return (source.a_acc(t).dot(a_vel) / n_a_vel) * normalize(a_vel)

    @staticmethod
    def a_acc_n(t: float) -> np.ndarray:
        source = Task1
        return source.a_acc(t) - source.a_acc_t(t)

    @staticmethod
    def b_coord(t: float) -> np.ndarray:
        source = Task1

        val = source.check_buffered(source.BUFFERED_B_COORDS, t)
        if isinstance(val, np.ndarray):
            return val

        a_coord = source.a_coord(t)
        o2_coord = source.FIXED_POINTS_COORDS[1]
        eq = source.create_eq(
            a_coord, o2_coord, source.AB, source.BO2,
            lambda point: point[1] < -source.C
        )
        sol = fsolve(
            eq,
            source.POSS_B_COORD
        )
        source.POSS_B_COORD = np.array((sol[0], sol[1]))
        source.BUFFERED_B_COORDS[t] = np.array((sol[0], sol[1], 0))
        return source.BUFFERED_B_COORDS[t]

    @staticmethod
    def b_vel(t: float) -> np.ndarray:
        source = Task1
        return (source.b_coord(t + source.DT) - source.b_coord(t)) / source.DT

    @staticmethod
    def b_acc(t: float) -> np.ndarray:
        source = Task1
        return (source.b_vel(t + source.DT) - source.b_vel(t)) / source.DT

    @staticmethod
    def b_acc_t(t: float) -> np.ndarray:
        source = Task1
        b_vel = source.b_vel(t)
        n_b_vel = np.linalg.norm(b_vel)
        return (source.b_acc(t).dot(b_vel) / n_b_vel) * normalize(b_vel)

    @staticmethod
    def b_acc_n(t: float) -> np.ndarray:
        source = Task1
        return source.b_acc(t) - source.b_acc_t(t)

    @staticmethod
    def c_coord(t: float) -> np.ndarray:
        source = Task1
        a_coord = source.a_coord(t)
        b_coord = source.b_coord(t)
        return a_coord + source.AC * source.C_ROT_MATRIX.dot((b_coord - a_coord) / source.AB)

    @staticmethod
    def c_vel(t: float) -> np.ndarray:
        source = Task1
        a_vel = source.a_vel(t)
        b_vel = source.b_vel(t)
        return a_vel + source.AC * source.C_ROT_MATRIX.dot((b_vel - a_vel) / source.AB)

    @staticmethod
    def d_coord(t: float) -> np.ndarray:
        source = Task1
        c_coord = source.c_coord(t)
        return np.array(
            (
                sqrt(source.CD ** 2 - (source.D - c_coord[1]) ** 2) + c_coord[0],
                source.D,
                0
            )
        )

    @staticmethod
    def d_vel(t: float) -> np.ndarray:
        source = Task1
        c_coord = source.c_coord(t)
        c_vel = source.c_vel(t)
        return np.array(
            (
                c_vel[0] + (source.D - c_coord[1]) * c_vel[1] / sqrt(source.CD ** 2 - (source.D - c_coord[1]) ** 2),
                0,
                0
            )
        )

    @staticmethod
    def e_coord(t: float) -> np.ndarray:
        source = Task1
        c_coord = source.c_coord(t)
        d_coord = source.d_coord(t)
        return (source.CE / source.CD) * (d_coord - c_coord) + c_coord

    @staticmethod
    def e_vel(t: float) -> np.ndarray:
        source = Task1
        c_vel = source.c_vel(t)
        d_vel = source.d_vel(t)
        return (source.CE / source.CD) * (d_vel - c_vel) + c_vel

    @staticmethod
    def f_coord(t: float) -> np.array:
        source = Task1

        val = source.check_buffered(source.BUFFERED_F_COORDS, t)
        if isinstance(val, np.ndarray):
            return val

        e_coord = source.e_coord(t)
        o3_coord = source.FIXED_POINTS_COORDS[2]

        sol = fsolve(
            source.create_eq(
                e_coord, o3_coord, source.EF, source.FO3,
                lambda point: point[1] * (source.A + source.B) > point[0] * (source.D + source.E)
            ),
            source.POSS_F_COORD
        )
        source.POSS_F_COORD = np.array((sol[0], sol[1]))
        source.BUFFERED_F_COORDS[t] = np.array((sol[0], sol[1], 0))
        return source.BUFFERED_F_COORDS[t]

    @staticmethod
    def f_vel(t: float) -> np.ndarray:
        source = Task1
        return (source.f_coord(t + source.DT) - source.f_coord(t)) / source.DT

    # Creates equation of intersection of two circles within a given filter
    @staticmethod
    def create_eq(a_coord: np.ndarray, b_coord: np.ndarray, r1: float, r2: float,
                  filter_f: Callable[[np.ndarray], bool]):
        return lambda point: (
            (point[0] - a_coord[0]) ** 2 + (point[1] - a_coord[1]) ** 2 - r1 ** 2,
            (point[0] - b_coord[0]) ** 2 + (point[1] - b_coord[1]) ** 2 - r2 ** 2
        ) if not filter_f(point) else (10 ** 6, 10 ** 6)

    # Check if coordinate already have calculated
    @staticmethod
    def check_buffered(buffered: dict, t: float) -> TUnion[np.ndarray, bool]:
        if t in buffered:
            return buffered[t]

        if Task1.CRITICAL_TIME_1 < t < Task1.CRITICAL_TIME_2:
            buffered[t] = np.array((0, 0, 0))
            return buffered[t]

        return False

    @staticmethod
    def body_d_line(x: float) -> float:
        return Task1.D

    # Creates law of angular velocity between to points (A, B) and distance law, d_law
    @staticmethod
    def get_ang_vel_law(a_law: Callable[[float], np.ndarray],
                        b_law: Callable[[float], np.ndarray],
                        d_law: Callable[[float], np.ndarray]) -> Callable[[float], float]:
        return lambda ct: sign(
            np.cross(d_law(ct), b_law(ct) - a_law(ct))[2]
        ) * np.linalg.norm(b_law(ct) - a_law(ct)) / np.linalg.norm(d_law(ct))

    def get_all_fixed_points(self, axes: Axes):
        return (
            Dot(
                c2p(axes, point),
                color=self.POINT_COLOR,
                radius=self.POINT_RADIUS
            )
            for point in self.FIXED_POINTS_COORDS
        )

    def get_all_mut_points(self, t: ValueTracker, axes: Axes):
        laws = [self.a_coord, self.b_coord, self.c_coord, self.d_coord, self.e_coord, self.f_coord]
        return (
            get_mut_dot(t, axes, self.POINT_COLOR, self.POINT_RADIUS, law)
            for law in laws
        )

    def get_all_lines(self, t: ValueTracker, axes: Axes):
        laws = [
            (lambda ct: self.FIXED_POINTS_COORDS[0], self.a_coord),
            (lambda ct: self.FIXED_POINTS_COORDS[1], self.b_coord),
            (lambda ct: self.FIXED_POINTS_COORDS[2], self.f_coord),
            (self.c_coord, self.d_coord),
            (self.e_coord, self.f_coord),
        ]
        return (
            get_line(t, axes, GREEN, s_law, e_law, stroke_width=DEFAULT_STROKE_WIDTH / 2)
            for s_law, e_law in laws
        )

    def get_all_vel(self, t: ValueTracker, axes: Axes):
        s_functions = [self.a_coord, self.b_coord, self.c_coord, self.d_coord, self.e_coord, self.f_coord]
        e_functions = [self.a_vel, self.b_vel, self.c_vel, self.d_vel, self.e_vel, self.f_vel]
        c = BLUE
        scale = 0.5
        return (
            get_vector(t, axes, c, s_fun, e_fun, scale, self.VECTORS_KWARGS)
            for s_fun, e_fun in zip(s_functions, e_functions)
        )

    def get_all_acc(self, t: ValueTracker, axes: Axes):
        s_functions = [self.a_coord, self.b_coord] * 3
        e_functions = [self.a_acc, self.b_acc, self.a_acc_t, self.b_acc_t, self.a_acc_n, self.b_acc_n]
        colors = [RED] * 2 + [YELLOW] * 4
        scale = 0.2
        return (
            get_vector(t, axes, c, s_fun, e_fun, scale, self.VECTORS_KWARGS)
            for c, s_fun, e_fun in zip(colors, s_functions, e_functions)
        )

    def get_points_titles(self, t: ValueTracker, axes: Axes):
        laws = [
            lambda ct: self.FIXED_POINTS_COORDS[0],
            lambda ct: self.FIXED_POINTS_COORDS[1],
            lambda ct: self.FIXED_POINTS_COORDS[2],
            self.a_coord, self.b_coord, self.c_coord, self.d_coord, self.e_coord, self.f_coord
        ]
        shift_laws = [
                         lambda ct: np.array((-0.15, -0.15, 0))
                     ] * len(laws)
        titles = ['O_1', 'O_2', 'O_3', 'A', 'B', 'C', 'D', 'E', 'F']
        font_size = 22
        return (
            get_dot_title(t, axes, title, font_size, law, shift_law)
            for title, law, shift_law in zip(titles, laws, shift_laws)
        )

    @staticmethod
    def get_velocities_titles(t: ValueTracker, velocities: list[Arrow]):
        shift_laws = [
                         lambda vec: normalize(vec) * 0.2
                     ] * len(velocities)
        titles = ['\\vec{V}_A', '\\vec{V}_B', '\\vec{V}_C', '\\vec{V}_D', '\\vec{V}_E', '\\vec{V}_F']
        font_size = 22
        return (
            get_vector_title(t, vector, title, font_size, shift_law)
            for vector, title, shift_law in zip(velocities, titles, shift_laws)
        )

    @staticmethod
    def get_acc_titles(t: ValueTracker, accelerations: list[Arrow]):
        shift_laws = [
                         lambda vec: normalize(vec) * 0.2
                     ] * len(accelerations)

        shift_laws[0] = lambda vec: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).dot(normalize(vec) * 0.2)
        shift_laws[2] = lambda vec: np.array((0.25, 0, 0))
        shift_laws[3] = lambda vec: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).dot(normalize(vec) * 0.2)
        shift_laws[4] = lambda vec: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).dot(normalize(vec) * 0.2)
        shift_laws[5] = lambda vec: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).dot(normalize(vec) * 0.2)

        titles = ['\\vec{a}_A', '\\vec{a}_B',
                  '\\vec{a}_{\\tau,A}', '\\vec{a}_{\\tau,B}',
                  '\\vec{a}_{n,A}', '\\vec{a}_{n,B}']
        font_size = 22
        return (
            get_vector_title(t, vector, title, font_size, shift_law)
            for vector, title, shift_law in zip(accelerations, titles, shift_laws)
        )

    @staticmethod
    def get_point_trajectory(axes: Axes, law: Callable[[float], np.ndarray], t_range: np.ndarray,
                             num_dashes: int):
        curve = axes.plot_parametric_curve(
            law, color=WHITE, t_range=t_range,
            stroke_width=DEFAULT_STROKE_WIDTH / 5
        )
        return DashedVMobject(curve, num_dashes=num_dashes)

    # Below functions related with graph
    @staticmethod
    def set_graph_labels(axes: Axes, graphs: list[TUnion[ParametricFunction, VGroup]],
                         labels: list[str], font_size: int, directions: list[np.ndarray], configs: dict):
        g_labels = []
        for graph, label, direction in zip(graphs, labels, directions):
            g_label = None
            configs['direction'] = direction

            if isinstance(graph, ParametricFunction):
                g_label = axes.get_graph_label(
                    graph, MathTex(label, font_size=font_size), **configs
                )
            elif isinstance(graph, VGroup):
                g_label = axes.get_graph_label(
                    graph.submobjects[1], MathTex(label, font_size=font_size), **configs
                )
            g_labels.append(g_label)
        return VGroup(*g_labels)

    @staticmethod
    def get_len_fun(law):
        return lambda ct: np.linalg.norm(law(ct))

    def get_plots_of_all_velocities(self, axes: Axes):
        laws = [self.a_vel, self.b_vel, self.c_vel, self.d_vel, self.e_vel, self.f_vel]
        colors = [BLUE, RED, GREEN, YELLOW, PURPLE, WHITE]

        return VGroup(
            *
            [
                axes.plot(self.get_len_fun(laws[0]), [0, self.TIME_INTERVALS[-1], self.DT / 3], stroke_width=1,
                          color=colors[0])
            ] + [
                VGroup(
                    axes.plot(self.get_len_fun(law), [0, self.TIME_INTERVALS[1], self.DT / 3], stroke_width=1, color=c),
                    axes.plot(self.get_len_fun(law), [self.TIME_INTERVALS[2], self.TIME_INTERVALS[-1], self.DT / 3],
                              stroke_width=1, color=c)
                )
                for law, c in zip(laws[1:], colors[1:])
            ]
        )

    def get_plots_of_all_ang_vel(self, axes: Axes):
        laws = [
            self.get_ang_vel_law(
                lambda ct: np.array((0, 0, 0)),
                self.a_vel,
                lambda ct: self.a_coord(ct) - self.FIXED_POINTS_COORDS[0]
            ),
            self.get_ang_vel_law(
                self.a_vel,
                self.b_vel,
                lambda ct: self.b_coord(ct) - self.a_coord(ct)
            ),
            self.get_ang_vel_law(
                lambda ct: np.array((0, 0, 0)),
                self.b_vel,
                lambda ct: self.b_coord(ct) - self.FIXED_POINTS_COORDS[1]
            ),
            self.get_ang_vel_law(
                self.c_vel,
                self.d_vel,
                lambda ct: self.d_coord(ct) - self.c_coord(ct)
            ),
            self.get_ang_vel_law(
                self.f_vel,
                self.e_vel,
                lambda ct: self.e_coord(ct) - self.f_coord(ct)
            ),
            self.get_ang_vel_law(
                lambda ct: np.array((0, 0, 0)),
                self.f_vel,
                lambda ct: self.f_coord(ct) - self.FIXED_POINTS_COORDS[2]
            ),
        ]
        colors = [BLUE, RED, GREEN, YELLOW, PURPLE, WHITE]

        return VGroup(
            *
            [
                axes.plot(laws[0], [0, self.TIME_INTERVALS[-1], self.DT / 3], stroke_width=1, color=colors[0])
            ] + [
                VGroup(
                    axes.plot(law, [0, self.TIME_INTERVALS[1], self.DT / 3], stroke_width=1, color=c),
                    axes.plot(law, [self.TIME_INTERVALS[2], self.TIME_INTERVALS[-1], self.DT / 3],
                              stroke_width=1, color=c)
                )
                for law, c in zip(laws[1:], colors[1:])
            ]
        )

    def get_plots_of_all_acc(self, axes: Axes):
        laws = [self.a_acc, self.b_acc]
        colors = [BLUE, RED]

        return VGroup(
            *
            [
                axes.plot(self.get_len_fun(laws[0]), [0, self.TIME_INTERVALS[-1], self.DT / 2], stroke_width=1,
                          color=colors[0])
            ] + [
                VGroup(
                    axes.plot(self.get_len_fun(law), [0, self.TIME_INTERVALS[1], self.DT / 2], stroke_width=1, color=c),
                    axes.plot(self.get_len_fun(law), [self.TIME_INTERVALS[2], self.TIME_INTERVALS[-1], self.DT / 2],
                              stroke_width=1, color=c)
                )
                for law, c in zip(laws[1:], colors[1:])
            ]
        )

    def get_plot_for_ang_acc(self, axes: Axes):
        ang_vel_law = self.get_ang_vel_law(
            self.a_vel,
            self.b_vel,
            lambda ct: self.b_coord(ct) - self.a_coord(ct)
        )
        return VGroup(
            axes.plot(
                lambda ct: (ang_vel_law(ct + self.DT) - ang_vel_law(ct)) / self.DT,
                [0, self.TIME_INTERVALS[1]], stroke_width=1, color=BLUE
            ),
            axes.plot(
                lambda ct: (ang_vel_law(ct + self.DT) - ang_vel_law(ct)) / self.DT,
                [self.TIME_INTERVALS[2], self.TIME_INTERVALS[-1]], stroke_width=1, color=BLUE
            ),
        )

    # Entities
    axes = None
    t = None
    o1_point, o2_point, o3_point = [None] * 3
    o1_title, o2_title, o3_title = [None] * 3
    a_point, b_point, c_point, d_point, e_point, f_point = [None] * 6
    a_title, b_title, c_title, d_title, e_title, f_title = [None] * 6
    a_p_vel, b_p_vel, c_p_vel, d_p_vel, e_p_vel, f_p_vel = [None] * 6
    a_p_acc, b_p_acc, a_p_acc_t, b_p_acc_t, a_p_acc_n, b_p_acc_n = [None] * 6
    a_t_vel, b_t_vel, c_t_vel, d_t_vel, e_t_vel, f_t_vel = [None] * 6
    a_t_acc, b_t_acc, a_t_acc_t, b_t_acc_t, a_t_acc_n, b_t_acc_n = [None] * 6
    ao1_line, bo2_line, fo3_line, cd_line, ef_line = [None] * 5
    triangle = None

    def show_graphs(self):
        v_configs = {
            'x_range': [0, self.TIME_INTERVALS[-1] + 0.2, 0.2],
            'y_range': [0, 70, 5],
            'x_length': 3.5,
            'y_length': 3.5,
            'axis_config': {
                'stroke_width': 1,
                'tip_width': 0.1,
                'tip_height': 0.1,
                'tick_size': 0.02,
                'line_to_number_buff': MED_SMALL_BUFF / 4,
                'font_size': 11,
                'decimal_number_config': {
                    'num_decimal_places': 1,
                },
            },
            'x_axis_config': {
                'numbers_to_include': np.arange(0, self.TIME_INTERVALS[-1] + 0.2, 0.2),
            },
            'y_axis_config': {
                'numbers_to_include': np.arange(0, 70, 5),
                'decimal_number_config': {
                    'num_decimal_places': 0,
                }
            },
        }

        a_configs = copy.deepcopy(v_configs)
        a_configs['y_range'] = [-4, 4, 1]
        a_configs['y_axis_config']['numbers_to_include'] = np.arange(-3.5, 4, 0.5)
        a_configs['y_axis_config']['decimal_number_config']['num_decimal_places'] = 1

        acc_configs = copy.deepcopy(v_configs)
        acc_configs['y_range'] = [0, 200, 20]
        acc_configs['y_axis_config']['numbers_to_include'] = np.arange(0, 200, 20)

        a_acc_configs = copy.deepcopy(v_configs)
        a_acc_configs['y_range'] = [-3.5, 3.5, 0.5]
        a_acc_configs['y_axis_config']['numbers_to_include'] = np.arange(-3.5, 3.5, 0.5)
        a_acc_configs['y_axis_config']['decimal_number_config']['num_decimal_places'] = 1

        v_axes = Axes(**v_configs).move_to(UL * 1.9 + LEFT)
        v_axes_labels = v_axes.get_axis_labels(
            MathTex('t,s', font_size=22), MathTex('V, \\frac{m}{s}', font_size=22)
        )

        ang_axes = Axes(**a_configs).move_to(UR * 1.9 + RIGHT)
        ang_axes_labels = ang_axes.get_axis_labels(
            MathTex('t,s', font_size=22), MathTex('\\omega, \\frac{1}{s}', font_size=22)
        )

        acc_axes = Axes(**acc_configs).move_to(DL * 1.9 + LEFT)
        acc_axes_labels = acc_axes.get_axis_labels(
            MathTex('t,s', font_size=22), MathTex('a, \\frac{m}{s^2}', font_size=22)
        )
        acc_axes_labels.submobjects[1].shift(DOWN * 0.35)

        a_acc_axes = Axes(**a_acc_configs).move_to(DR * 1.9 + RIGHT)
        a_acc_axes_labels = a_acc_axes.get_axis_labels(
            MathTex('t,s', font_size=22), MathTex('\\epsilon, \\frac{1}{s^2}', font_size=22)
        )
        a_acc_axes_labels.submobjects[1].shift(DOWN * 0.35)

        configs = {
            'buff': MED_SMALL_BUFF / 3,
            'direction': RIGHT,
            'dot': True,
            'x_val': self.TIME_INTERVALS[-1],
            'dot_config': {'radius': DEFAULT_DOT_RADIUS / 4},
        }

        vel_plots = self.get_plots_of_all_velocities(v_axes)
        v_graph_labels = self.set_graph_labels(
            v_axes,
            graphs=vel_plots.submobjects,
            labels=['V_A', 'V_B', 'V_C', 'V_D', 'V_E', 'V_F'],
            font_size=22,
            directions=[UL, UR, RIGHT, UR, DR, RIGHT],
            configs=configs
        )

        ang_plots = self.get_plots_of_all_ang_vel(ang_axes)
        ang_graph_labels = self.set_graph_labels(
            ang_axes,
            graphs=ang_plots.submobjects,
            labels=[
                '\\omega_{AO_1}', '\\omega_{BA}=\\omega_{CA}=\\omega_{BC}',
                '\\omega_{BO_2}', '\\omega_{DC}', '\\omega_{EF}', '\\omega_{FO_3}'
            ],
            font_size=22,
            directions=[UL, UP + UR, RIGHT, DR, UR, RIGHT],
            configs=configs
        )

        acc_plots = self.get_plots_of_all_acc(acc_axes)
        acc_graph_labels = self.set_graph_labels(
            acc_axes,
            graphs=acc_plots.submobjects,
            labels=['a_A', 'a_B'],
            font_size=22,
            directions=[RIGHT] * 2,
            configs=configs
        )

        a_acc_plots = self.get_plot_for_ang_acc(a_acc_axes)
        a_acc_graph_labels = self.set_graph_labels(
            a_acc_axes,
            graphs=a_acc_plots.submobjects,
            labels=['\\epsilon_{BA}'],
            font_size=22,
            directions=[UR],
            configs=configs
        )

        self.add(
            v_axes, v_axes_labels, v_graph_labels, vel_plots,
            ang_axes, ang_axes_labels, ang_plots, ang_graph_labels,
            acc_axes, acc_axes_labels, acc_plots, acc_graph_labels,
            a_acc_axes, a_acc_axes_labels, a_acc_plots, a_acc_graph_labels,
        )

    def construct1(self):
        initial_time = self.TIME_INTERVALS[0]
        end_time = self.TIME_INTERVALS[1]

        self.t = ValueTracker(initial_time)

        self.axes = Axes(**self.AXES_CONFIG)
        axes_labels = self.axes.get_axis_labels()

        body_d_line = DashedVMobject(
            self.axes.plot(self.body_d_line, [-35, 115], stroke_width=1, color=BLUE),
            num_dashes=30
        )

        self.o1_point, self.o2_point, self.o3_point = self.get_all_fixed_points(self.axes)
        self.a_point, self.b_point, self.c_point, self.d_point, self.e_point, self.f_point = \
            self.get_all_mut_points(self.t, self.axes)

        self.ao1_line, self.bo2_line, self.fo3_line, self.cd_line, self.ef_line = self.get_all_lines(self.t, self.axes)

        self.a_p_vel, self.b_p_vel, self.c_p_vel, self.d_p_vel, self.e_p_vel, self.f_p_vel = \
            self.get_all_vel(self.t, self.axes)

        self.a_p_acc, self.b_p_acc, self.a_p_acc_t, self.b_p_acc_t, self.a_p_acc_n, self.b_p_acc_n = \
            self.get_all_acc(self.t, self.axes)

        self.o1_title, self.o2_title, self.o3_title, self.a_title, self.b_title, self.c_title, \
            self.d_title, self.e_title, self.f_title = self.get_points_titles(self.t, self.axes)

        self.a_t_vel, self.b_t_vel, self.c_t_vel, self.d_t_vel, self.e_t_vel, self.f_t_vel = \
            self.get_velocities_titles(
                self.t, [self.a_p_vel, self.b_p_vel, self.c_p_vel, self.d_p_vel, self.e_p_vel, self.f_p_vel]
            )

        self.a_t_acc, self.b_t_acc, self.a_t_acc_t, self.b_t_acc_t, self.a_t_acc_n, self.b_t_acc_n = \
            self.get_acc_titles(
                self.t, [self.a_p_acc, self.b_p_acc, self.a_p_acc_t, self.b_p_acc_t, self.a_p_acc_n, self.b_p_acc_n]
            )

        self.triangle = create_mut_obj(
            lambda tracker: Polygon(
                c2p(self.axes, self.a_coord(tracker.get_value())),
                c2p(self.axes, self.b_coord(tracker.get_value())),
                c2p(self.axes, self.c_coord(tracker.get_value())),
                color=GREEN, close_new_points=True, stroke_width=DEFAULT_STROKE_WIDTH / 2,
                fill_color=GREEN, fill_opacity=0.5
            ),
            self.t,
        )

        timer_pos = (LEFT + UP) * 3.5
        timer = get_vec_info(
            self.t, 't=', 22, lambda c_t: c_t, {'num_decimal_places': 3, 'font_size': 22}
        ).arrange(RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER / 5).move_to(timer_pos)

        curv_c = self.get_point_trajectory(
            self.axes, self.c_coord,
            np.array([self.TIME_INTERVALS[2], self.TIME_INTERVALS[3] + self.TIME_INTERVALS[1]]), 15
        )
        curv_e = self.get_point_trajectory(
            self.axes, self.e_coord,
            np.array([self.TIME_INTERVALS[2], self.TIME_INTERVALS[3] + self.TIME_INTERVALS[1]]), 15
        )

        self.add(
            self.axes,
            curv_c, curv_e,
            axes_labels, body_d_line,
            self.triangle, self.ao1_line, self.bo2_line, self.fo3_line, self.cd_line, self.ef_line,
            self.a_p_vel, self.b_p_vel, self.c_p_vel, self.d_p_vel, self.e_p_vel, self.f_p_vel,
            self.a_p_acc, self.b_p_acc, self.a_p_acc_t, self.b_p_acc_t, self.a_p_acc_n, self.b_p_acc_n,
            self.o1_point, self.o2_point, self.o3_point,
            self.a_point, self.b_point, self.c_point, self.d_point, self.e_point, self.f_point,
            self.o1_title, self.o2_title, self.o3_title,
            self.a_title, self.b_title, self.c_title, self.d_title, self.e_title, self.f_title,
            self.a_t_vel, self.b_t_vel, self.c_t_vel, self.d_t_vel, self.e_t_vel, self.f_t_vel,
            self.a_t_acc, self.b_t_acc, self.a_t_acc_t, self.b_t_acc_t, self.a_t_acc_n, self.b_t_acc_n,
            timer
        )
        self.play(self.t.animate.set_value(end_time), run_time=end_time - initial_time, rate_func=linear)

    def construct2(self):
        initial_time = self.TIME_INTERVALS[1]
        end_time = self.TIME_INTERVALS[2]

        self.t.set_value(initial_time)
        self.remove(
            self.bo2_line, self.fo3_line, self.cd_line, self.ef_line, self.triangle,
            self.b_point, self.c_point, self.d_point, self.e_point, self.f_point,
            self.b_p_vel, self.c_p_vel, self.d_p_vel, self.e_p_vel, self.f_p_vel,
            self.b_title, self.c_title, self.d_title, self.e_title, self.f_title,
            self.b_t_vel, self.c_t_vel, self.d_t_vel, self.e_t_vel, self.f_t_vel,
            self.b_p_acc,
            self.b_p_acc_t, self.b_p_acc_n,
            self.b_t_acc, self.b_t_acc_t, self.b_t_acc_n,
        )
        self.play(self.t.animate.set_value(end_time), run_time=end_time - initial_time, rate_func=linear)

    def construct3(self):
        initial_time = self.TIME_INTERVALS[2]
        end_time = self.TIME_INTERVALS[3]

        self.t.set_value(initial_time)
        self.remove(
            self.a_point, self.o1_point, self.o2_point, self.o3_point,
            self.o1_title, self.o2_title, self.o3_title, self.a_title,
            self.a_t_vel,
            self.a_p_acc,
            self.a_p_acc_t, self.a_p_acc_n,
            self.a_t_acc, self.a_t_acc_t, self.a_t_acc_n
        )
        self.add(
            self.triangle, self.bo2_line, self.fo3_line, self.cd_line, self.ef_line,
            self.a_p_vel, self.b_p_vel, self.c_p_vel, self.d_p_vel, self.e_p_vel, self.f_p_vel,
            self.a_p_acc, self.b_p_acc, self.a_p_acc_t, self.b_p_acc_t, self.a_p_acc_n, self.b_p_acc_n,
            self.o1_point, self.o2_point, self.o3_point,
            self.a_point, self.b_point, self.c_point, self.d_point, self.e_point, self.f_point,
            self.o1_title, self.o2_title, self.o3_title,
            self.a_title, self.b_title, self.c_title, self.d_title, self.e_title, self.f_title,
            self.a_t_vel, self.b_t_vel, self.c_t_vel, self.d_t_vel, self.e_t_vel, self.f_t_vel,
            self.a_t_acc, self.b_t_acc, self.a_t_acc_t, self.b_t_acc_t, self.a_t_acc_n, self.b_t_acc_n,
        )
        self.play(self.t.animate.set_value(end_time), run_time=end_time - initial_time, rate_func=linear)

    def construct(self):
        # First, animate
        self.construct1()
        self.construct2()
        self.construct3()

        # Then, plot graphs
        self.remove(*self.mobjects)
        self.play(Write(Text('Plots', font_size=80)))
        self.wait()
        self.remove(*self.mobjects)
        self.show_graphs()
        self.wait(2)
