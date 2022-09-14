from manim import *
from typing import Callable, Iterable, Optional, Union as TUnion


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def sign(v: float) -> float:
    if v > 0:
        return 1
    return -1


def c2p(axes: Axes, point: np.ndarray) -> np.ndarray:
    return axes.c2p(point[0], point[1], point[2])


def update_matrix(m: Matrix, values: Iterable):
    for mob, value in zip(m.get_entries(), values):
        mob.set_value(value)


def create_mut_obj(getter: Callable[[ValueTracker], Any],
                   tracker: ValueTracker,
                   updater: Callable = lambda getter, tracker: (lambda z: z.become(getter(tracker)))):
    obj = getter(tracker)
    obj.add_updater(updater(getter, tracker))
    return obj


def _get_vector(start: np.ndarray, end: np.ndarray,
                c: str, scale: float = 1.0, **kwargs) -> Arrow:
    return Arrow(
        start,
        start + (end - start) * scale,
        color=c,
        **kwargs
    )


def get_vector(t: ValueTracker, axes: Axes, c: str,
               start_function: Callable[[float], np.ndarray],
               end_function: Callable[[float], np.ndarray],
               scale: float, a_kwargs: dict = None) -> Arrow:
    res: Arrow = create_mut_obj(
        lambda tracker: _get_vector(
            c2p(axes, start_function(tracker.get_value())),
            c2p(axes, start_function(tracker.get_value()) + end_function(tracker.get_value())),
            c,
            scale,
            **a_kwargs if a_kwargs else dict()
        ),
        t
    )
    return res


def get_line(t: ValueTracker, axes: Axes, c: str,
             s_law: Callable[[float], np.ndarray], e_law: Callable[[float], np.ndarray],
             **kwargs):
    res: Line = create_mut_obj(
        lambda tracker: Line(
            c2p(axes, s_law(tracker.get_value())),
            c2p(axes, e_law(tracker.get_value())),
            buff=0,
            color=c,
            **kwargs
        ),
        t,
    )
    return res


def get_vector_title(t: ValueTracker, vector: Arrow, title: str, font_size: int,
                     shift_law: Callable[[np.ndarray], np.ndarray]) -> MathTex:
    res: MathTex = create_mut_obj(
        lambda tracker: MathTex(title, font_size=font_size).move_to(
            vector.get_end() + shift_law(vector.get_end() - vector.get_start())
        ),
        t,
        lambda getter, tracker:
        lambda z: z.move_to(vector.get_end() + shift_law(vector.get_end() - vector.get_start()))
    )
    return res


def get_dot_title(t: ValueTracker, axes: Axes, title: str, font_size: int,
                  law: Callable[[float], np.ndarray], shift_law: Callable[[float], np.ndarray]) -> MathTex:
    res: MathTex = create_mut_obj(
        lambda tracker: MathTex(title, font_size=font_size).move_to(
            c2p(axes, law(tracker.get_value())) + shift_law(tracker.get_value())
        ),
        t,
        lambda getter, tracker:
        lambda z: z.move_to(c2p(axes, law(tracker.get_value())) + shift_law(tracker.get_value()))
    )
    return res


def get_mut_dot(t: ValueTracker, axes: Axes, c: str, radius: float, law: Callable[[float], np.ndarray]) -> Dot:
    return create_mut_obj(
        lambda tracker: Dot(
            c2p(axes, law(tracker.get_value())),
            color=c,
            radius=radius
        ),
        t,
        lambda getter, tracker: lambda z: z.move_to(c2p(axes, law(tracker.get_value())))
    )


def _get_vec_val(t: ValueTracker, data: Callable[[float], TUnion[np.ndarray, float]],
                 data_configs: Optional[dict] = None, **kwargs) -> TUnion[DecimalMatrix, DecimalNumber]:
    data_configs = data_configs if data_configs else dict()
    initial_data = data(t.get_value())
    scale = kwargs.get('scale', 1.0)

    if isinstance(initial_data, np.ndarray):
        return create_mut_obj(
            lambda tracker: DecimalMatrix(
                [[scalar] for scalar in initial_data],
                **data_configs
            ).scale(scale),
            t,
            lambda getter, tracker: lambda z: update_matrix(z, data(tracker.get_value()))
        )
    else:
        return create_mut_obj(
            lambda tracker: DecimalNumber(
                initial_data,
                **data_configs
            ),
            t,
            lambda getter, tracker: lambda z: z.set_value(data(tracker.get_value()))
        )


def get_vec_info(t: ValueTracker, label: str, font_size: int,
                 data: Callable[[float], TUnion[np.ndarray, float]],
                 data_configs: Optional[dict] = None, **kwargs) -> VGroup:
    return VGroup(
        MathTex(label, font_size=font_size),
        _get_vec_val(t, data, data_configs, **kwargs)
    )


def cut(law: Callable[[float], np.ndarray]):
    return lambda c_t: law(c_t)[:-1]


def deep_arrange(group: VGroup, direction: np.ndarray, buff: float, center: bool, aligned_edge: np.ndarray):
    group.arrange(direction, buff, center, aligned_edge=aligned_edge)
    for sub1, sub2 in zip(group.submobjects, group.submobjects[1:]):
        for m1, m2 in zip(sub1.submobjects[1:], sub2.submobjects[1:]):
            m2.next_to(m1, direction, buff, aligned_edge=aligned_edge)
    return group
