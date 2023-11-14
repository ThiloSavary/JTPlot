import re
from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from typing import TypedDict, Tuple, Literal, List

from matplotlib.patches import ConnectionPatch

target = 'ANG'
solvens = 'DMSO'


class SingleEnergyDisplay:
    y_ranges = None
    states = {}
    background_color = None
    ax1 = None
    ax2 = None
    fig = None
    energy_lines = {}
    energy_width = 0.6

    def __init__(
            self,
            y_ranges: Tuple[Tuple[float, float], Tuple[float, float]],
            background_color: str = '#00000000',
            states: Tuple[str, ...] = ('S0', 'S1', 'T1'),
            figsize: Tuple[float, float] = None,
            dpi: int = None
    ):
        self.background_color = background_color

        x_ticks = []
        for i, state in enumerate(states):
            self.states[state] = [i + 1 - self.energy_width / 2, i + 1 + self.energy_width / 2]
            x_ticks.append(i + 1)

        y_ranges = [(y_ranges[0][0] - 0.2, y_ranges[0][1]), (y_ranges[1][0], y_ranges[1][1] + 0.2)]
        self.y_ranges = y_ranges

        total_height = sum(y[1] - y[0] for y in y_ranges)
        height_ratios = [(y[1] - y[0]) / total_height for y in y_ranges]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=height_ratios, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(hspace=0.05, wspace=0)

        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax2.xaxis.tick_bottom()
        ax1.tick_params(labeltop=False)

        ax1.set_ylim(y_ranges[0])
        ax2.set_ylim(y_ranges[1])

        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(states)
        ax2.set_xlim(0.5, len(states) + 0.5)

        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig

    def __get_axis__(self, y_val):
        if self.y_ranges[0][0] < y_val < self.y_ranges[0][1]:
            return self.ax1
        elif self.y_ranges[1][0] < y_val < self.y_ranges[1][1]:
            return self.ax2
        raise ValueError(f'Energy {y_val} is not in the range of the y axes.')

    def add_energy(
            self, state: str,
            energy: float,
            color: str = 'black',
            label: str = None,

            text_label: str = None,
            text_size: float = None,
            text_va: Literal['center', 'bottom', 'top'] = 'center',
            text_ha: Literal['center', 'left', 'right'] = 'left'
    ):
        if label is None:
            n = len([x for x in self.energy_lines.keys() if x.startswith(state)])
            label = f'{state}_{n}'

        # Check if label is already used
        for key in self.energy_lines.keys():
            if label == key:
                raise ValueError(f'Label {label} is already used.')

        self.energy_lines[label] = {
            'state': state,
            'energy': energy,
        }

        ax = self.__get_axis__(energy)
        ax.plot(self.states[state], [energy, energy], color=color)
        if text_label is not None:
            ax.text(self.states[state][1] + 0.02, energy,
                    text_label, color=color, va=text_va, ha=text_ha, size=text_size)

        return label

    def connect_energies(
            self,
            label1: str,
            label2: str,
            color: str = 'black',

            line_style: str = '--',

            text: str = None,
            text_offset: Tuple[float, float] = (0, 0),
            text_color: str = None,
            text_size: float = None,
            text_va: Literal['center', 'bottom', 'top'] = 'bottom',
            text_ha: Literal['center', 'left', 'right'] = 'center',
            rotate_text: bool = True,
            arrow_style: str = '-',

            line_width: float = None
    ):
        if label1 not in self.energy_lines.keys():
            raise ValueError(f'Label {label1} is not used.')
        if label2 not in self.energy_lines.keys():
            raise ValueError(f'Label {label2} is not used.')

        energy_line1 = self.energy_lines[label1]
        energy_line2 = self.energy_lines[label2]

        # Check if energies have the same state
        if energy_line1['state'] == energy_line2['state']:
            xy1 = (
                (self.states[energy_line1['state']][0] + self.states[energy_line1['state']][1]) / 2,
                energy_line1['energy']
            )
            xy2 = (
                (self.states[energy_line2['state']][0] + self.states[energy_line2['state']][1]) / 2,
                energy_line2['energy']
            )

            self.add_line(xy1, xy2, color=color, line_style=line_style, arrow_style=arrow_style,
                          text=text, text_offset=text_offset, text_color=text_color, text_ha=text_ha,
                          text_size=text_size, text_va=text_va, rotate_text=rotate_text, line_width=line_width)

            return

        # Check if state of energy_line1 is lower than state of energy_line2
        if self.states[energy_line1['state']][0] > self.states[energy_line2['state']][0]:
            # Swap energy_line1 and energy_line2
            energy_line1, energy_line2 = energy_line2, energy_line1

        xy1 = (self.states[energy_line1['state']][1], energy_line1['energy'])
        xy2 = (self.states[energy_line2['state']][0], energy_line2['energy'])

        self.add_line(xy1, xy2, color=color, line_style=line_style, arrow_style=arrow_style,
                      text=text, text_offset=text_offset, text_color=text_color, text_ha=text_ha,
                      text_size=text_size, text_va=text_va, rotate_text=rotate_text, line_width=line_width)

    def add_line(
            self,
            xy1: Tuple[float, float],
            xy2: Tuple[float, float],
            color: str = 'black',

            line_style: str = '--',

            text: str = None,
            text_offset: Tuple[float, float] = (0, 0),
            text_color: str = None,
            text_size: float = None,
            text_va: Literal['center', 'bottom', 'top'] = 'bottom',
            text_ha: Literal['center', 'left', 'right'] = 'center',
            rotate_text: bool = True,
            arrow_style: str = '-',

            line_width: float = None
    ):
        ax1 = self.__get_axis__(xy1[1])
        ax2 = self.__get_axis__(xy2[1])
        con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA=ax1.transData, coordsB=ax2.transData, color=color,
                              linestyle=line_style, arrowstyle=arrow_style, linewidth=line_width)

        self.fig.add_artist(con)

        if text is not None:
            xy = np.mean([ax1.transData.transform(xy1), ax2.transData.transform(xy2)], axis=0)

            if rotate_text:
                angle = 180 / np.pi * np.arctan2(*(ax2.transData.transform(xy2) - ax1.transData.transform(xy1))[::-1])
            else:
                angle = 0

            distance = text_offset
            xytext = (
                np.cos(np.deg2rad(angle)) * distance[0] - np.sin(np.deg2rad(angle)) * distance[1],
                np.sin(np.deg2rad(angle)) * distance[0] + np.cos(np.deg2rad(angle)) * distance[1]
            )

            if text_color is None:
                text_color = color

            plt.annotate(text, xy, xycoords='figure pixels', va=text_va, ha=text_ha,
                         color=text_color, rotation=angle, rotation_mode='anchor', textcoords='offset pixels',
                         xytext=xytext, size=text_size)

    def finalize(self):
        fig = self.fig
        ax1 = self.ax1
        ax2 = self.ax2

        x0, y0 = fig.transFigure.inverted().transform((ax2.transAxes.transform((0, 0))))
        x1, _ = fig.transFigure.inverted().transform((ax2.transAxes.transform((1, 0))))
        _, y1 = fig.transFigure.inverted().transform((ax1.transAxes.transform((1, 1))))

        rect = patches.Rectangle(
            xy=(x0, y0),
            width=x1 - x0,
            height=y1 - y0,
            transform=fig.transFigure,
            clip_on=False,
            color=self.background_color,
            zorder=1,
            linewidth=1
        )
        fig.add_artist(rect)


class Geometry(TypedDict):
    name: str
    singlet: Tuple[float, float, float]  # GS, 1st excited state, 2nd excited state
    triplet: Tuple[float, float]  # 1st excited state, 2nd excited state


class GeometryReal(TypedDict):
    name: str
    singlet: Tuple[float, float, float]  # 1st excited state, 2nd excited state
    triplet: Tuple[float, float]  # 1st excited state, 2nd excited state


def to_eV(energy: float) -> float:
    return energy * 27.2114


def calculate_real_geometry(geometry: Geometry, ground_state: float) -> GeometryReal:
    singlet = (to_eV(abs(geometry['singlet'][0] - ground_state)), to_eV(abs(geometry['singlet'][1] - ground_state)),
               to_eV(abs(geometry['singlet'][2] - ground_state)))
    triplet = (to_eV(abs(geometry['triplet'][0] - ground_state)), to_eV(abs(geometry['triplet'][1] - ground_state)))
    return GeometryReal(singlet=singlet, triplet=triplet, name=geometry['name'])


def plotit(geometries: List[GeometryReal]):
    lower_max = max([x['singlet'][0] for x in geometries])
    upper_min = 0.9 * min(
        [x['singlet'][1] for x in geometries] +
        [x['singlet'][2] for x in geometries] +
        [x['triplet'][0] for x in geometries] +
        [x['triplet'][1] for x in geometries]
    )
    upper_max = 1.1 * max(
        [x['singlet'][1] for x in geometries] +
        [x['singlet'][2] for x in geometries] +
        [x['triplet'][0] for x in geometries] +
        [x['triplet'][1] for x in geometries]
    )

    y_ranges = ((upper_min, upper_max), (-0.2, lower_max))

    states = ('S0', 'S1', 'T1')

    display = SingleEnergyDisplay(y_ranges=y_ranges, states=states, dpi=600)

    def add_stair(getter, label, color):
        keys = []
        for i, geometry in enumerate(geometries):
            name = geometry['name']

            if i == len(geometry) - 1:
                key = display.add_energy(name, getter(geometry), text_label=label, color=color)
            else:
                key = display.add_energy(name, getter(geometry), color=color)
            keys.append(key)

        for i in range(1, len(keys)):
            display.connect_energies(keys[i - 1], keys[i], color=color)

        print(keys)

    add_stair(lambda geometry: geometry['singlet'][0], label=None, color='black')
    add_stair(lambda geometry: geometry['singlet'][1], label='S$_1$', color='blue')
    add_stair(lambda geometry: geometry['singlet'][2], label='S$_2$', color='blue')
    add_stair(lambda geometry: geometry['triplet'][0], label='T$_1$', color='#7F00FF')
    add_stair(lambda geometry: geometry['triplet'][1], label='T$_2$', color='#7F00FF')

    try:
        data = get_wavelength_and_oscilator_st_for('S0')

        display.connect_energies('S0_0', 'S0_1',
                                 line_style='-',
                                 text=f'{data[0][:-1]} nm\n\n$f = {round(float(data[1]), 3)}$',
                                 text_va='center',
                                 text_offset=(0, 0),  # Change this to move the text (-20, 0) lowers text
                                 arrow_style="-|>",
                                 line_width=2)

    except IndexError:
        pass

    try:
        data = get_wavelength_and_oscilator_st_for('S1')

        display.connect_energies('S1_0', 'S1_1',
                                 color='blue',
                                 line_style='-',
                                 text=f'{data[0][:-1]} nm\n\n$f = {round(float(data[1]), 3)}$',
                                 text_va='center',
                                 text_offset=(0, 0),  # Change this to move the text (-20, 0) lowers text
                                 arrow_style="<|-",
                                 line_width=2)
    except IndexError:
        pass

    # Get S1 Geometry
    s1 = next(x for x in geometries if x['name'] == 'S1')
    # Get T1 Geometry
    t1 = next(x for x in geometries if x['name'] == 'T1')

    val = round(abs(s1['singlet'][1] - t1['triplet'][0]), 3)
    display.connect_energies('S1_1', 'T1_3',
                             line_style='-',
                             arrow_style="<->",
                             color='red',
                             text_va='top',
                             text_ha='left',
                             text_offset=(10, -20),
                             text=f'$\\Delta E_{{ST}}= {val}$ eV',
                             line_width=2)

    display.finalize()
    plt.savefig('plot.png')


def get_path():
    path = Path(__file__).parent / target
    if solvens == 'Vacuum':
        path = path / "Vacuum"
    else:
        path = path / "Solvens" / solvens
    return path


def read_data_to_geometries() -> List[Geometry]:
    path = get_path()
    path = path / "optDFT"

    states = ('S0', 'S1', 'T1')

    geometries = []
    for state in states:
        state_path_singlet = path / state / "DFTMRCI" / "singlet.in.sum"
        state_path_triplet = path / state / "DFTMRCI" / "triplet.in.sum"

        if not state_path_singlet.exists():
            print(f"File {state_path_singlet} does not exist.")
            continue

        with open(state_path_singlet, "r") as file:
            text = file.read()
        try:
            s0 = re.findall(r'#\s+1\s+1a\s+DFTCI\s+(-?\d+.\d+)', text)[0]
            s1 = re.findall(r'#\s+2\s+2a\s+DFTCI\s+(-?\d+.\d+)', text)[0]
            s2 = re.findall(r'#\s+3\s+3a\s+DFTCI\s+(-?\d+.\d+)', text)[0]

        except IndexError:
            print(f"Could not find energy in {state_path_singlet}")
            continue

        if not state_path_triplet.exists():
            print(f"File {state_path_triplet} does not exist.")
            print("Trying to find triplet folder ...")

            state_path_triplet = path / state / "DFTMRCI" / "triplet" / "triplet.in.sum"

            if not state_path_triplet.exists():
                print(f"File {state_path_triplet} does not exist.")
                continue

        with open(state_path_triplet, "r") as file:
            text = file.read()
        try:
            t1 = re.findall(r'#\s+1\s+1a\s+DFTCI\s+(-?\d+.\d+)', text)[0]
            t2 = re.findall(r'#\s+2\s+2a\s+DFTCI\s+(-?\d+.\d+)', text)[0]
        except IndexError:
            print(f"Could not find energy in {state_path_triplet}")
            continue

        geometries.append({
            'name': state,
            'singlet': (float(s0), float(s1), float(s2)),
            'triplet': (float(t1), float(t2))
        })

    return geometries


# Die Werte die am Pfeil stehen ...
def get_wavelength_and_oscilator_st_for(state):
    path = get_path()
    path = path / "optDFT"

    with open(path / state / "DFTMRCI" / "singlet.in_autoref.dmat.1a.prp", "r") as file:
        text = file.read()
    try:
        return re.findall(r'2\s+\d+\w+\s+->\d+\w+\s+\d+.\d+\s+(\d+.\d*)\s+\d+.\d*\s+(\d+.\d*)', text)[0]
    except IndexError as error:
        print(
            f"Could not find wavelength and oscillator strength in {path / state / 'DFTMRCI' / 'singlet.in_autoref.dmat.1a.prp'}")
        raise IndexError from error


if __name__ == '__main__':
    _raw_geometries = read_data_to_geometries()

    assert len(_raw_geometries) > 0, "No data found."

    print(_raw_geometries)

    # Find Geometry with name 'S0'
    _s0 = next(x for x in _raw_geometries if x['name'] == 'S0')

    _ground_state = _s0['singlet'][0]

    geometries = [calculate_real_geometry(x, _ground_state) for x in _raw_geometries]

    plotit(geometries)
