﻿from pypairing import PyFq, PyFq2, PyFq12, PyFqRepr, PyG1, PyG2, PyFr
import random
import re
from inspect import getframeinfo, stack

# Order of BLS group
bls12_381_r = 52435875175126190479447740508185965837690552500527637822603658699938581184513  # (# noqa: E501)


def dupe_pyg1(pyg1):
    out = PyG1()
    out.copy(pyg1)
    return out


def dupe_pyg2(pyg2):
    out = PyG2()
    out.copy(pyg2)
    return out


def dupe_pyfr(pyfr):
    out = PyFr("1")
    out.copy(pyfr)
    return out


def dupe_pyfq12(pyfq12):
    out = PyFq12("1")
    out.copy(pyfq12)
    return out


class G1:
    def __init__(self, other=None):
        if other is None:
            self.pyg1 = PyG1()
        if type(other) is list:
            assert len(other) == 2
            assert len(other[0]) == 6
            x = PyFqRepr(other[0][0], other[0][1], other[0][2],
                         other[0][3], other[0][4], other[0][5])
            y = PyFqRepr(other[1][0], other[1][1], other[1][2],
                         other[1][3], other[1][4], other[1][5])
            xq = PyFq()
            yq = PyFq()
            xq.from_repr(x)
            yq.from_repr(y)
            self.pyg1 = PyG1()
            self.pyg1.load_fq_affine(xq, yq)
        elif type(other) is PyG1:
            self.pyg1 = other

    def __str__(self):
        return self.pyg1.__str__()
        # caller = getframeinfo(stack()[1][0])
        # print("%s:%d" % (caller.filename, caller.lineno))
        # x = int(self.pyg1.__str__()[4:102], 0)
        # y = int(self.pyg1.__str__()[108:206], 0)
        # return "(" + str(x) + ", " + str(y) + ")"

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        if type(other) is G1:
            out = dupe_pyg1(self.pyg1)
            out.add_assign(other.pyg1)
            return G1(out)
        else:
            raise TypeError(
                'Invalid multiplication param. Expected G1. Got '
                + str(type(other)))

    def __imul__(self, other):
        if type(other) is G1:
            self.pyg1.add_assign(other.pyg1)
            return self
        raise TypeError(
            'Invalid multiplication param. Expected G1. Got '
            + str(type(other)))

    def __truediv__(self, other):
        if type(other) is G1:
            out = dupe_pyg1(self.pyg1)
            out.sub_assign(other.pyg1)
            return G1(out)
        else:
            raise TypeError(
                'Invalid division param. Expected G1. Got '
                + str(type(other)))

    def __idiv__(self, other):
        if type(other) is G1:
            self.pyg1.sub_assign(other.pyg1)
            return self
        raise TypeError(
            'Invalid division param. Expected G1. Got '
            + str(type(other)))

    def __pow__(self, other):
        if type(other) is int:
            out = G1(dupe_pyg1(self.pyg1))
            if other == 0:
                out.pyg1.zero()
                return out
            if other < 0:
                out.pyg1.negate()
                other *= -1
            out.pyg1.mul_assign(ZR(other).val)
            return out
        elif type(other) is ZR:
            out = G1(dupe_pyg1(self.pyg1))
            out.pyg1.mul_assign(other.val)
            return out
        else:
            raise TypeError(
                'Invalid exponentiation param. Expected ZR or int. Got '
                + str(type(other)))

    def __ipow__(self, other):
        if type(other) is int:
            if other == 0:
                self.pyg1.zero()
                return self
            if other < 0:
                self.invert()
                other *= -1
            self.pyg1.mul_assign(ZR(other).val)
            return self
        elif type(other) is ZR:
            self.pyg1.mul_assign(other.val)
            return self
        else:
            raise TypeError(
                'Invalid exponentiation param. Expected ZR or int. Got '
                + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __rpow__(self, other):
        return self.__pow__(other)

    def __eq__(self, other):
        if type(other) is not G1:
            return False
        return self.pyg1.equals(other.pyg1)

    def __getstate__(self):
        coords = self.pyg1.__str__()
        x = coords[6:102]
        y = coords[110:206]
        xlist = [x[80:96], x[64:80], x[48:64], x[32:48], x[16:32], x[0:16]]
        ylist = [y[80:96], y[64:80], y[48:64], y[32:48], y[16:32], y[0:16]]
        for i in range(6):
            xlist[i] = int(xlist[i], 16)
            ylist[i] = int(ylist[i], 16)
        return [xlist, ylist]

    def __setstate__(self, d):
        self.__init__(d)

    def invert(self):
        negone = PyFr(str(1))
        negone.negate()
        self.pyg1.mul_assign(negone)

    def duplicate(self):
        return G1(dupe_pyg1(self.pyg1))

    def projective(self):
        return self.pyg1.projective()

    def pair_with(self, other):
        fq12 = PyFq12()
        self.pyg1.py_pairing_with(other.pyg2, fq12)
        return GT(fq12)

    @staticmethod
    def one():
        one = G1()
        one.pyg1.zero()
        return one

    @staticmethod
    def rand(seed=None):
        out = PyG1()
        if seed is None:
            seed = []
            for _ in range(4):
                seed.append(random.SystemRandom().randint(0, 4294967295))
            out.rand(seed[0], seed[1], seed[2], seed[3])
        else:
            assert type(seed) is list
            assert len(seed) == 4
            out.rand(seed[0], seed[1], seed[2], seed[3])
        return G1(out)


class G2:
    def __init__(self, other=None):
        if other is None:
            self.pyg2 = PyG2()
        if type(other) is list:
            assert len(other) == 4
            assert len(other[0]) == 6
            x1 = PyFqRepr(other[0][0], other[0][1], other[0][2],
                          other[0][3], other[0][4], other[0][5])
            x2 = PyFqRepr(other[1][0], other[1][1], other[1][2],
                          other[1][3], other[1][4], other[1][5])
            y1 = PyFqRepr(other[2][0], other[2][1], other[2][2],
                          other[2][3], other[2][4], other[2][5])
            y2 = PyFqRepr(other[3][0], other[3][1], other[3][2],
                          other[3][3], other[3][4], other[3][5])
            xq = PyFq2()
            yq = PyFq2()
            xq.from_repr(x1, x2)
            yq.from_repr(y1, y2)
            self.pyg2 = PyG2()
            self.pyg2.load_fq_affine(xq, yq)
        elif type(other) is PyG2:
            self.pyg2 = other

    def __str__(self):
        out = self.pyg2.__str__()
        x1 = int(out[8:106], 0)
        x2 = int(out[113:211], 0)
        y1 = int(out[226:324], 0)
        y2 = int(out[331:429], 0)
        return "(" + str(x1) + " + " + str(x2) + "u, " + str(y1) + " + " + str(y2) + "u)"

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        if type(other) is G2:
            out = dupe_pyg2(self.pyg2)
            out.add_assign(other.pyg2)
            return G2(out)

    def __imul__(self, other):
        if type(other) is G2:
            self.pyg2.add_assign(other.pyg2)
            return self

    def __truediv__(self, other):
        if type(other) is G2:
            out = dupe_pyg2(self.pyg2)
            out.sub_assign(other.pyg2)
            return G2(out)
        else:
            raise TypeError(
                'Invalid division param. Expected G1. Got '
                + str(type(other)))

    def __idiv__(self, other):
        if type(other) is G2:
            self.pyg2.sub_assign(other.pyg2)
            return self

    def __pow__(self, other):
        if type(other) is int:
            out = G2(dupe_pyg2(self.pyg2))
            if other == 0:
                out.pyg2.zero()
                return out
            if other < 0:
                out.pyg2.negate()
                other *= -1
            out.pyg2.mul_assign(ZR(other).val)
            return out
        elif type(other) is ZR:
            out = G2(dupe_pyg2(self.pyg2))
            out.pyg2.mul_assign(other.val)
            return out
        else:
            raise TypeError(
                'Invalid exponentiation param. Expected ZR or int. Got '
                + str(type(other)))

    def __ipow__(self, other):
        if type(other) is int:
            if other == 0:
                self.pyg2.zero()
                return self
            if other < 0:
                self.invert()
                other *= -1
            self.pyg2.mul_assign(ZR(other).val)
            return self
        elif type(other) is ZR:
            self.pyg2.mul_assign(other.val)
            return self
        else:
            raise TypeError(
                'Invalid exponentiation param. Expected ZR or int. Got '
                + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __rpow__(self, other):
        return self.__pow__(other)

    def __eq__(self, other):
        if type(other) is not G2:
            return False
        return self.pyg2.equals(other.pyg2)

    def __getstate__(self):
        coords = self.pyg2.__str__()
        x = coords[6:102]
        y = coords[110:206]
        xlist = [x[80:96], x[64:80], x[48:64], x[32:48], x[16:32], x[0:16]]
        ylist = [y[80:96], y[64:80], y[48:64], y[32:48], y[16:32], y[0:16]]
        for i in range(6):
            xlist[i] = int(xlist[i], 16)
            ylist[i] = int(ylist[i], 16)
        return [xlist, ylist]

    def __setstate__(self, d):
        self.__init__(d)

    def invert(self):
        negone = PyFr(str(1))
        negone.negate()
        self.pyg2.mul_assign(negone)

    def duplicate(self):
        return G2(dupe_pyg2(self.pyg2))

    def projective(self):
        return self.pyg2.projective()

    @staticmethod
    def one():
        one = G2()
        one.pyg2.zero()
        return one

    @staticmethod
    def rand(seed=None):
        out = PyG2()
        if seed is None:
            seed = []
            for _ in range(4):
                seed.append(random.SystemRandom().randint(0, 4294967295))
            out.rand(seed[0], seed[1], seed[2], seed[3])
        else:
            assert type(seed) is list
            assert len(seed) == 4
            out.rand(seed[0], seed[1], seed[2], seed[3])
        return G2(out)


class GT:
    def __init__(self, other=None):
        if other is None:
            self.pyfq12 = PyFq12()
            self.pyfq12.rand(1, 0, 0, 0)
        elif type(other) is PyFq12:
            self.pyfq12 = other
        elif type(other) is list:
            assert len(other) == 12
            self.pyfq12 = PyFq12()
            self.pyfq12.from_strs(*other)
        elif type(other) is int:
            self.pyfq12 = PyFq12()
            self.pyfq12.from_strs(str(other), *["0"]*11)
        elif type(other) is str:
            lst = [x.strip() for x in other.split(',')]
            assert len(lst) == 12
            if lst[0][1] == 'x':
                for i in range(len(lst)):
                    lst[i] = str(int(lst[i]))
            self.pyfq12 = PyFq12()
            self.pyfq12.from_strs(*lst)

    def __mul__(self, other):
        if type(other) is int:
            out = GT(dupe_pyfq12(self.pyfq12))
            if other == 0:
                out.pyfq12.zero()
                return out
            if other < 0:
                out.pyfq12.negate()
                other *= -1
            prodend = GT(other)
            out.pyfq12.mul_assign(prodend.pyfq12)
            return out
        elif type(other) is ZR:
            out = GT(dupe_pyfq12(self.pyfq12))
            out.pyfq12.mul_assign(other.val)
            return out
        elif type(other) is GT:
            out = GT(dupe_pyfq12(self.pyfq12))
            out.pyfq12.mul_assign(other.pyfq12)
            return out
        else:
            raise TypeError(
                'Invalid exponentiation param. Expected ZR or int. Got '
                + str(type(other)))

    def __add__(self, other):
        if type(other) is GT:
            out = dupe_pyfq12(self.pyfq12)
            out.add_assign(other.pyfq12)
            return G2(out)

    # TODO: __truediv__ needs to be implemented

    # TODO: __pow__ needs to be implemented

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __rpow__(self, other):
        return self.__pow__(other)

    def __str__(self):
        out = self.pyfq12.__str__()
        return out

    def __eq__(self, other):
        if type(other) is not GT:
            return False
        return self.pyfq12.equals(other.pyfq12)

    def __getstate__(self):
        s = self.pyfq12.__str__()
        s = s.replace("Fq6", "")
        s = s.replace("Fq2", "")
        s = s.replace("Fq", "")
        s = s.replace(" ", "")
        s = s.replace("*", "")
        s = s.replace("u", "")
        s = s.replace("v^2", "")
        s = s.replace("v", "")
        s = s.replace("w", "")
        s = s.replace("(", "")
        s = s.replace(")", "")
        s = s.replace("+", ",")
        return re.sub("0x0*", "0x0", s)

    def __setstate__(self, d):
        self.__init__(d)

    @staticmethod
    def rand(seed=None):
        out = PyFq12()
        if seed is None:
            seed = []
            for _ in range(4):
                seed.append(random.SystemRandom().randint(0, 4294967295))
            out.rand(seed[0], seed[1], seed[2], seed[3])
        else:
            assert type(seed) is list
            assert len(seed) == 4
            out.rand(seed[0], seed[1], seed[2], seed[3])
        return GT(out)


class ZR:
    def __init__(self, val):
        self.pp = []
        # if val is None:
        #     self.val = PyFr(0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654)
        # elif type(val) is int:
        # assert type(val) is int
        self.val = PyFr(str(abs(val)))
        if val < 0:
            self.val.negate()
        # elif type(val) is str:
        #     if val[1] == 'x':
        #         self.val = PyFr(val)
        #     elif int(val) < 0:
        #         intval = int(val) * -1
        #         self.val = PyFr(str(intval))
        #         self.val.negate()
        #     else:
        #         self.val = PyFr(val)
        # elif type(val) is PyFr:
        #     self.val = val

    def __str__(self):
        hexstr = self.val.__str__()[3:-1]
        return str(int(hexstr, 0))

    def __repr__(self):
        return str(self)

    def __int__(self):
        hexstr = self.val.__str__()[3:-1]
        return int(hexstr, 0)

    def __add__(self, other):
        if type(other) is ZR:
            out = dupe_pyfr(self.val)
            out.add_assign(other.val)
            return ZR(out)
        elif type(other) is int:
            out = dupe_pyfr(self.val)
            if other < 0:
                other *= -1
                addend = PyFr(str(other))
                addend.negate()
            else:
                addend = PyFr(str(other))
            out.add_assign(addend)
            return ZR(out)
        else:
            raise TypeError(
                'Invalid addition param. Expected ZR or int. Got '
                + str(type(other)))

    def __radd__(self, other):
        assert type(other) is int
        return self.__add__(ZR(other))

    def __iadd__(self, other):
        self.pp = []
        if type(other) is ZR:
            self.val.add_assign(other.val)
            return self
        elif type(other) is int:
            if other < 0:
                other *= -1
                addend = PyFr(str(other))
                addend.negate()
            else:
                addend = PyFr(str(other))
            self.val.add_assign(addend)
            return self
        else:
            raise TypeError(
                'Invalid addition param. Expected ZR or int. Got '
                + str(type(other)))

    def __sub__(self, other):
        if type(other) is ZR:
            out = dupe_pyfr(self.val)
            out.sub_assign(other.val)
            return ZR(out)
        elif type(other) is int:
            out = dupe_pyfr(self.val)
            if other < 0:
                other *= -1
                subend = PyFr(str(other))
                subend.negate()
            else:
                subend = PyFr(str(other))
            out.sub_assign(subend)
            return ZR(out)
        else:
            raise TypeError(
                'Invalid addition param. Expected ZR or int. Got '
                + str(type(other)))

    def __rsub__(self, other):
        assert type(other) is int
        return ZR(other).__sub__(self)

    def __isub__(self, other):
        self.pp = []
        if type(other) is ZR:
            self.val.sub_assign(other.val)
            return self
        elif type(other) is int:
            if other < 0:
                other *= -1
                subend = PyFr(str(other))
                subend.negate()
            else:
                subend = PyFr(str(other))
            self.val.sub_assign(subend)
            return self
        else:
            raise TypeError(
                'Invalid addition param. Expected ZR or int. Got '
                + str(type(other)))

    def __mul__(self, other):
        if type(other) is ZR:
            out = dupe_pyfr(self.val)
            out.mul_assign(other.val)
            return ZR(out)
        elif type(other) is int:
            out = dupe_pyfr(self.val)
            if other < 0:
                other *= -1
                prodend = PyFr(str(other))
                prodend.negate()
            else:
                prodend = PyFr(str(other))
            out.mul_assign(prodend)
            return ZR(out)
        else:
            raise TypeError(
                'Invalid multiplication param. Expected ZR or int. Got '
                + str(type(other)))

    def __imul__(self, other):
        self.pp = []
        if type(other) is ZR:
            self.val.mul_assign(other.val)
            return self
        elif type(other) is int:
            if other < 0:
                other *= -1
                prodend = PyFr(str(other))
                prodend.negate()
            else:
                prodend = PyFr(str(other))
            self.val.mul_assign(prodend)
            return self
        else:
            raise TypeError(
                'Invalid multiplication param. Expected ZR or int. Got '
                + str(type(other)))

    def __rmul__(self, other):
        assert type(other) is int
        return self.__mul__(ZR(other))

    def __truediv__(self, other):
        if type(other) is ZR:
            out = dupe_pyfr(self.val)
            div = dupe_pyfr(other.val)
            div.inverse()
            out.mul_assign(div)
            return ZR(out)
        elif type(other) is int:
            out = dupe_pyfr(self.val)
            if other < 0:
                other *= -1
                prodend = PyFr(str(other))
                prodend.negate()
            else:
                prodend = PyFr(str(other))
            prodend.inverse()
            out.mul_assign(prodend)
            return ZR(out)
        else:
            raise TypeError(
                'Invalid division param. Expected ZR or int. Got '
                + str(type(other)))

    def __pow__(self, other):
        if type(other) is int:
            other = other % (bls12_381_r-1)
        elif type(other) is ZR:
            other = int(other)
        else:
            raise TypeError(
                'Invalid multiplication param. Expected int or ZR. Got '
                + str(type(other)))
        if other == 0:
            return ZR(1)
        out = dupe_pyfr(self.val)
        if self.pp == []:
            self.init_pp()
        i = 0
        # Hacky solution to my off by one error
        other -= 1
        while other > 0:
            if other % 2 == 1:
                out.mul_assign(self.pp[i])
            i += 1
            other = other >> 1
        return ZR(out)

    def __eq__(self, other):
        if type(other) is int:
            other = ZR(other)
        assert type(other) is ZR
        return self.val.equals(other.val)

    def __getstate__(self):
        return int(self)

    def __setstate__(self, d):
        self.__init__(d)

    def init_pp(self):
        self.pp.append(dupe_pyfr(self.val))
        for i in range(1, 255):
            power = dupe_pyfr(self.pp[i-1])
            power.square()
            self.pp.append(power)

    @staticmethod
    def random(seed=None):
        r = bls12_381_r
        if seed is None:
            r = random.randint(0, r-1)
            return ZR(r)
        else:
            # Generate pseudorandomly based on seed
            r = random.Random(seed).randint(0, r-1)
            return ZR(r)

    @staticmethod
    def zero():
        return ZR(PyFr("0"))

    @staticmethod
    def one():
        return ZR(PyFr("1"))
