from manim import *
from manim import config
import pandas as pd
import numpy as np
import operator
from typing import Callable

SCALE_FACTOR = 4  # Manifold scaling factor
FPS = config.frame_rate  # fps, either 15 or 60 for ql or normal
TRAIL_LENGTH = FPS * 1  # Length of trail behind mercury
FIRST_ORBIT_TIME = 4  # Desired length of first orbit
REST_OF_ORBIT_TIME = 3  # Desired length of sped up animation
POINTS_PER_ORBIT = int(90023 / (2 / 0.2408467))  # 90023 points = 2 yrs -> 0.2408467 yrs = 1 rev
YEARS = 2000  # Number of years of mercurys orbit to be animated
ORBITS = YEARS / 0.2408467  # Number of revolutions, can be float (# yrs / # yrs/rev)


def prepare_data(file_name: str) -> pd.DataFrame:
    """
    Prepare Data Method

    Scales data to fit on screen

    :param file_name: The name of the CSV file containing the data.
    :return: The processed dataframe.

    """
    orbit_points = int(POINTS_PER_ORBIT * ORBITS)  # 90023 points = 5 yrs -> 0.2408467 yrs = 1 rev

    df = pd.read_csv(file_name, nrows=orbit_points, usecols=[1, 2, 3, 8], dtype=np.float64)
    df.reset_index(drop=True, inplace=True)
    max_value = df['RG'].values.max()

    for col in ['X', 'Y', 'Z']:
        df[col] = df[col] / max_value * SCALE_FACTOR
    return df


# Taken from plot_advancement.py, did not import for versitility in case modifications are needed
def get_peri_indices(ds):
    """
    Grabs periapsis vector indices from a list of distances from the center coord in an elliptical orbit
    :param ds: arraylike of distances from center coord.
    :return: list of indices of peri-vectors
    """
    peri_indices = []
    for i, r in enumerate(ds):
        if i != 0 and i != len(ds) - 1:
            if ds.iloc[i - 1] > r < ds.iloc[i + 1]:
                peri_indices.append(i)
    return peri_indices


# noinspection PyUnusedLocal
def make_updater(angle_stop: float,
                 wiggle_tracker: ValueTracker,
                 speed: float,
                 oscillation_amplitude: float,
                 oscillation_speed: float,
                 vertical: int,
                 comparison: Callable) -> Callable:
    """
    Creates an updater func to adjust camera's angle and simulate oscillation.

    :param angle_stop: Desired angle to stop oscillation.
    :param wiggle_tracker: Tacker obj to control the oscillation.
    :param speed: Speed at which camera rotates.
    :param oscillation_amplitude: Amplitude of oscillation.
    :param oscillation_speed: Speed of oscillation.
    :param vertical: Whether oscillation is vertical.
    :param comparison: Function for determining direction of rotation.
    :return: Updater function.
    """

    def updater(angle_tracker: ValueTracker) -> None:
        if comparison(angle_tracker.get_value(), angle_stop) and wiggle_tracker.get_value() == 0:
            adjustment = (angle_stop - angle_tracker.get_value()) / abs(angle_stop - angle_tracker.get_value()) * speed
            return angle_tracker.set_value(angle_tracker.get_value() + adjustment)
        elif not comparison(angle_tracker.get_value(), angle_stop) and wiggle_tracker.get_value() == 0:
            return wiggle_tracker.increment_value(0.01)
        if wiggle_tracker.get_value() != 0:
            adjustment_to_angle = oscillation_amplitude * (
                    np.sin(wiggle_tracker.get_value() - (np.pi * vertical / 2)) - vertical)
            wiggle_tracker.increment_value(oscillation_speed)
            return angle_tracker.set_value(angle_stop + adjustment_to_angle)

    return updater


def points_to_parametric_function(pts, **kwargs):
    n = len(pts) - 1
    return ParametricFunction(lambda t: np.array(
        [pts[int(t * n)][0], pts[int(t * n)][1], pts[int(t * n)][2]]), **kwargs)


# noinspection PyUnresolvedReferences,PyUnusedLocal
class FadingPath(VMobject):
    """
    Custom VMobject subclass to create a fading path trail.
    Contains an updater method to update each of its points.
    """

    def __init__(self,
                 traced_mobject: Mobject,
                 trail_length: int = TRAIL_LENGTH,
                 stroke_width: int = SCALE_FACTOR,
                 stroke_color: ManimColor = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if stroke_color is None:
            stroke_color = [PURE_BLUE, PINK]
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.traced_mobject = traced_mobject
        self.trail_length = trail_length
        self.transparent_color = stroke_color[0] if isinstance(stroke_color, list) else stroke_color
        self.opaque_color = stroke_color[1] if isinstance(stroke_color, list) else stroke_color
        self.start_new_path(self.traced_mobject.get_center())
        self.add_updater(self.update_trace)

    def update_trace(self, mobj, dt):
        new_line = Line(self[-1].get_end(), self.traced_mobject.get_center())

        # calculate linear gradient function for opacity
        max_opacity = 1
        min_opacity = 0
        opacity_step = (max_opacity - min_opacity) / len(self)

        new_line.set_opacity(max_opacity)
        new_line.set_color(self.opaque_color)
        self.add(new_line)

        if len(self) > self.trail_length:
            for i, line in enumerate(self):
                if i in [0, 1]:  # For some reason, just removing last line doesn't work, so remove last two
                    self.remove(line)

        for i, line in enumerate(self):
            # update opacity for each line in the path group
            new_opacity = min_opacity + i * opacity_step
            line.set_stroke(opacity=new_opacity)

            # update color of each line in the path group
            ratio = (i / len(self)) ** 2
            new_color = interpolate_color(self.transparent_color, self.opaque_color, ratio)
            line.set_color(new_color)


# noinspection PyAttributeOutsideInit
class Orbit(ThreeDScene):
    """
    Class to create 3D visualization of Mercury's orbit.
    """

    def construct(self):
        axes = ThreeDAxes(num_axis_pieces=20,
                          color=GREY,
                          axis_config={
                              'stroke_opacity': 0.9,
                              'stroke_width': 0.7,
                              'include_tip': False
                          })

        self.camera.background_color = BLACK

        df = prepare_data('data/horizons.csv')  # This processes Mercury's orbit data

        # list of points, with each point of the form [x, y, z]
        points = df[['X', 'Y', 'Z']].values

        # Mercury representation
        mercury = Dot3D(radius=SCALE_FACTOR * 0.01, color=GREY)
        mercury.move_to(points[0])
        index_tracker = ValueTracker(0)  # Tracker for locus of mercury
        mercury.add_updater(lambda m: m.move_to(
            points[min(int(index_tracker.get_value()), len(points) - 1)]
        ))

        fading_path = FadingPath(mercury)

        # Sun representation (radius close to 4.375% of orbit)
        sun = Dot3D(radius=.6 * SCALE_FACTOR * 0.04375, color=ORANGE)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-135 * DEGREES)

        # initialize trackers for camera orientation
        phi: ValueTracker = self.camera.phi_tracker
        theta: ValueTracker = self.camera.theta_tracker

        # initiliaze camera related variables
        camera_speed = 0.0075 / (FPS/15)
        oscillation_speed = 0.77 / (FPS/15)
        oscillation_amplitude = 0.005
        left_right_tracker = ValueTracker(0)
        up_down_tracker = ValueTracker(0)
        gap_from_start = 10 * DEGREES
        phi_stop = phi.get_value() - gap_from_start
        theta_stop = theta.get_value() + gap_from_start

        phi.add_updater(
            make_updater(
                phi_stop, up_down_tracker, camera_speed,
                oscillation_amplitude, oscillation_speed, 1, comparison=operator.gt
            )
        )

        theta.add_updater(
            make_updater(
                theta_stop, left_right_tracker, camera_speed,
                oscillation_amplitude, oscillation_speed, 0, comparison=operator.lt
            )
        )

        self.add(axes, mercury, sun, fading_path, phi, theta)

        # Carefully animate one full orbit and some
        self.play(index_tracker.animate.set_value(int(POINTS_PER_ORBIT*1.2)), rate_func=linear, run_time=FIRST_ORBIT_TIME)

        # For some reason this is needed to prevent camera from going back to start
        index_tracker.animate.set_value(int(POINTS_PER_ORBIT * 1.2))

        # Rush through remaining orbits
        self.play(index_tracker.animate.set_value(len(points) - 1), rate_func=linear, run_time=REST_OF_ORBIT_TIME)

        mercury.clear_updaters()
        phi.clear_updaters()
        theta.clear_updaters()

        self.remove(mercury, fading_path)

        self.set_camera_orientation(phi=phi_stop, theta=theta_stop)

        # Approximately looking normal to orbit plane, found with mean anuglar momentum vector
        self.play(phi.animate.set_value(28.5 * DEGREES), theta.animate.set_value(-78.7 * DEGREES), run_time=1.5)

        start_orbit = points_to_parametric_function(points[:POINTS_PER_ORBIT], color=BLUE, stroke_width=0.4 * SCALE_FACTOR)
        end_orbit = points_to_parametric_function(points[-POINTS_PER_ORBIT:], color=RED, stroke_width=0.4 * SCALE_FACTOR)

        self.add(start_orbit, end_orbit)

        self.play(FadeIn(start_orbit, rate_func=lambda t: t * 0.8),
                  FadeIn(end_orbit, rate_func=lambda t: t * 0.8),
                  run_time=0.75)

        # Make perihelion vectors showing start and end orbit difference in perihelion
        start_i = get_peri_indices(df.iloc[:POINTS_PER_ORBIT]['RG'])[0]
        end_i = get_peri_indices(df.iloc[-POINTS_PER_ORBIT:]['RG'])[0]
        start_peri = np.array(df.iloc[start_i][['X', 'Y', 'Z']].values)
        end_peri = np.array(df.iloc[end_i - POINTS_PER_ORBIT][['X', 'Y', 'Z']].values)
        start_orbit_vector = Arrow3D(start=ORIGIN, end=start_peri, color=BLUE, thickness=.02, height=0.3, base_radius=0.04)
        end_orbit_vector = Arrow3D(start=ORIGIN, end=end_peri, color=RED, thickness=.02, height=0.3, base_radius=0.04)

        # Make angle arc and also display shift in arcsecs
        r = np.linalg.norm(start_peri)
        angle_arc = ArcBetweenPoints(start_peri, end_peri, radius=r, color=LIGHT_PINK)
        angle = angle_arc.angle * 180 / np.pi * 3600  # rad to arcsec
        angle_text = Text(f"Advancment: {angle:.1f} (as)", font_size=24, color=YELLOW).next_to(angle_arc, UP)

        # Display other info
        span_text = Text(f"Span: {YEARS} years", font_size=24, color=YELLOW).next_to(angle_text, UP)
        start_text = Text("Start", font_size=24, color=YELLOW).next_to(start_orbit_vector, RIGHT)
        end_text = Text("End", font_size=24, color=YELLOW).next_to(end_orbit_vector, LEFT)

        self.add(start_orbit_vector, end_orbit_vector, angle_arc, angle_text, span_text, start_text, end_text)

        self.play(FadeIn(start_orbit_vector),
                  FadeIn(end_orbit_vector),
                  FadeIn(angle_arc),
                  FadeIn(angle_text),
                  FadeIn(span_text),
                  FadeIn(start_text),
                  FadeIn(end_text),
                  run_time=0.75)

        self.remove(phi, theta)
        self.wait()
