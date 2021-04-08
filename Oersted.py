import numpy as np
import matplotlib.pyplot as plt
from typing import List
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numba
from numba.experimental import jitclass
BETA = 1

precision = numba.float32
element_spec = [('x0', precision), ('y0', precision), ('x1', precision),
                ('y1', precision), ('w', precision), ('h', precision),
                ('area', precision), ('_centre', precision[:]),
                ('I', precision), ('I_dir', precision[:])]


@jitclass(element_spec)
class Element:
    def __init__(self, x, y, width, height, I=1.0, I_dir=[0., 0., 1.]):
        self.x0 = x
        self.y0 = y
        self.x1 = x + width
        self.y1 = y + height

        self.w = width
        self.h = height

        self.area = self.w * self.h

        self.I = I
        self.I_dir = np.array(I_dir, dtype=np.float32)
        self._centre = np.asarray([(self.x1 + self.x0) / 2,
                                   (self.y1 + self.y0) / 2],
                                  dtype=np.float32)

    def get_contribution_at_p(self, p) -> float:
        dist = np.sqrt(
            np.sum(np.asarray([(p[i] - self._centre[i])**2 for i in (0, 1)])))
        return BETA * self.I / dist

    def get_vector_contribution_at_p(self, p) -> np.ndarray:
        diff_vector = p - self._centre
        perp_vector = np.cross(diff_vector, self.I_dir)
        dist = np.sqrt(np.sum(np.power(diff_vector, 2)))
        mag = BETA * self.I / dist
        return mag * perp_vector / np.linalg.norm(perp_vector)

    def set_current(self, I):
        self.I = I


class CompositeStructure:
    def __init__(self, composites, power=10):
        self.composites = composites
        # calculate the total resistance
        R_tot = sum([composite.resistance for composite in composites])
        I_tot = np.sqrt(power / R_tot)
        for composite in composites:
            rel_I = I_tot * R_tot / composite.resistance
            composite.set_per_element_current(rel_I)

    @staticmethod
    @numba.jit(nopython=False, parallel=True)
    def calculate_contribs(composites, Xs, Ys):
        total_contribs = np.zeros_like(Xs).astype(np.float32)
        for composite in composites:
            total_contribs += Composite.calculate_contribs(
                composite.elements, Xs.ravel(), Ys.ravel()).reshape(Xs.shape)
        return total_contribs


class Composite:
    def __init__(self, size, discretisation, offset, resistivity=1):
        """
        @param size:
            (W, L, H) aka (x, y, z)
        @param discretisation:
            sizes of tiny element
        """
        (self.w, self.h, self.l) = size
        self.area = self.w * self.h
        self.elements = mesh_discretisation(self.w,
                                            self.h,
                                            *discretisation,
                                            offset=offset)
        self.resistance = self.get_resistance(resistivity, self.l)
        self.offset = offset
        (self.Nx, self.Ny) = discretisation

    def set_current(self, I):
        element: Element
        for element in self.elements:
            element.set_current(I)

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def calculate_contribs(elements, Xs, Ys):
        vals = []
        for xp, yp in zip(Xs, Ys):
            p = (xp, yp)
            points_contributions = 0.0
            el: Element
            for el in elements:
                points_contributions += el.get_contribution_at_p(p)
            vals.append(points_contributions)
        return np.asarray(vals)

    @staticmethod
    @numba.jit(nopython=True, parallel=False)
    def calculate_vector_contribs(elements, Xs, Ys):
        vals = np.zeros((*Xs.shape, 2), dtype=np.float32)
        for i in range(Xs.shape[0]):
            for j in range(Xs.shape[1]):
                for el in elements:
                    vals[i, j, :] += el.get_vector_contribution_at_p(
                        np.asarray([Xs[i, j], Ys[i, j]]))
        return vals

    def get_resistance(self, resistivity, length):
        # assume all elements are of the same size
        return material_resistance(resistivity, self.area, self.l)

    def set_per_element_current(self, current):
        # the current is evenly distributed along the cross-sec.
        per_element_current = current / self.Nx * self.Ny
        self.set_current(per_element_current)
        return per_element_current


def mesh_discretisation(swidth, sheight, rwidth, rheight,
                        offset=(0, 0, 0)) -> List[Element]:
    """
    @param offset 
        top left corner (for the first block, it's usually (0,0,0) 
        i.e. origin point)
    Length => Y
    Height => Z
    Width => X
    """
    Nx, Ny = int(swidth / rwidth), int(sheight / rheight)
    (xoff, yoff) = offset
    element_groups: List[Element] = []
    for ix in range(Nx):
        for iy in range(Ny):
            element_groups.append(
                Element(xoff + ix * rwidth, yoff + iy * rheight, rwidth,
                        rheight))
    return element_groups


def material_resistance(resistivity, area, length):
    return resistivity * length / area


def generate_vector_field():
    sizeW = 40
    sizeL = 100
    sizeH = 90
    structure = [
        ("Co", 62.4e-9),
        ("Pt", 10.6e-8),
        ("Co", 62.4e-9),
    ]
    D = 4
    N = 3
    x = np.arange(-sizeW, (N + 1) * sizeW, D)
    y = np.arange(-sizeH, (N - 1) * sizeH, D)

    X, Y = np.meshgrid(x, y)

    composites = []
    CC = np.zeros_like(X).astype(float)
    VC = np.zeros((*X.shape, 2), dtype=np.float32)
    for i, (_, r) in enumerate(structure):
        composite = Composite(size=(sizeW, sizeH, sizeL),
                              discretisation=(D, D),
                              offset=(i * sizeW, 0),
                              resistivity=r)
        composite.set_current(1)
        VC += Composite.calculate_vector_contribs(composite.elements, X, Y)
        composites.append(composite)

    struct = CompositeStructure(composites)
    CC = CompositeStructure.calculate_contribs(struct.composites, X.ravel(),
                                               Y.ravel()).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(14, 14))
    _ = ax.pcolor(X[::2, ::2], Y[::2, ::2], CC[::2, ::2])

    im = ax.quiver(X[::2, ::2],
                   Y[::2, ::2],
                   VC[::2, ::2, 0],
                   VC[::2, ::2, 1],
                   units='width',
                   pivot='tip')
    ax.scatter(X[::2, ::2], Y[::2, ::2], color='r', s=0.5, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.tick_params(axis='x', rotation=90)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    _ = fig.colorbar(im, cax=cax, orientation='vertical')

    fig.savefig("Vector_field.png")

    for i, (name, _) in enumerate(structure):
        print(f"Layer {i}={name}")
        xoffset = i * sizeW
        yoffset = 0
        vv = get_vector_value_at_region(X, Y, VC, sizeW, sizeH, xoffset,
                                        yoffset, D)
        mag = np.linalg.norm(vv)
        print(f"\tMagnitude Hoe: {mag:.2f}")
        print(f"\tVector: {vv.tolist()}")


def get_vector_value_at_region(X, Y, VC, sizeW, sizeH, x0, y0, discretisation):
    d_width = int(sizeW / discretisation)
    d_height = int(sizeH / discretisation)

    # find starting blocks
    x_indx = np.abs(X[0, :] - x0).argmin()
    y_indx = np.abs(Y[0, :] - y0).argmin()

    print(x0, y0, x_indx, y_indx, d_width, d_height)
    return VC[x_indx:(x_indx + d_width),
              y_indx:(y_indx + d_height), :].sum(axis=(0, 1))


if __name__ == "__main__":
    generate_vector_field()
