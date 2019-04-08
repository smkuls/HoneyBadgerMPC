import operator
import logging
from functools import reduce
from .field import GF, GFElement
from itertools import zip_longest
from .betterpairing import ZR
from .elliptic_curve import Subgroup
from honeybadgermpc.ntl.helpers import fft as fft_cpp
from honeybadgermpc.ntl.helpers import fft_interpolate as fft_interpolate_cpp


def strip_trailing_zeros(a):
    if len(a) == 0:
        return []
    for i in range(len(a), 0, -1):
        if a[i-1] != 0:
            break
    return a[:i]


_poly_cache = {}


class PolyCache(object):
    _poly_cache = {}

    def __new__(cls, field):
        return PolyCache._poly_cache.setdefault(
            field, super(PolyCache, cls).__new__(cls))

    def __init__(self, field):
        self.field = field
        self._lagrange_cache = {}  # Cache lagrange polynomials
        self.field_type = GFElement if type(field) is GF else ZR

    def __call__(self, coefficients):
        return Polynomial(coefficients, self)

    def __reduce__(self):
        return (PolyCache, (self.field,))

    def interpolate_at(self, shares, x_recomb=None):
        x_recomb = x_recomb if x_recomb is not None else self.field(0)
        # shares are in the form (x, y=f(x))
        if type(x_recomb) is int:
            x_recomb = field(x_recomb)
        assert type(x_recomb) is self.field_type
        xs, ys = zip(*shares)
        vector = []
        for i, x_i in enumerate(xs):
            factors = [(x_k - x_recomb) / (x_k - x_i)
                       for k, x_k in enumerate(xs) if k != i]
            vector.append(reduce(operator.mul, factors))
        return sum(map(operator.mul, ys, vector))

    def interpolate(self, shares):
        # This is the polynomial f(x) = x
        x = Polynomial([self.field(0), self.field(1)], self)
        one = Polynomial([self.field(1)], self)  # This is the polynomial f(x) = 1
        xs, ys = zip(*shares)

        def lagrange(xi):
            # Let's cache lagrange values
            if (xs, xi) in self._lagrange_cache:
                return self._lagrange_cache[(xs, xi)]

            def mul(a, b): return a*b
            num = reduce(mul, [x - Polynomial([xj], self)
                               for xj in xs if xj != xi], one)
            den = reduce(mul, [xi - xj for xj in xs if xj != xi], self.field(1))
            p = num * Polynomial([1 / den], self)
            self._lagrange_cache[(xs, xi)] = p
            return p
        f = Polynomial([0], self)
        for xi, yi in zip(xs, ys):
            pi = lagrange(xi)
            f += Polynomial([yi], self) * pi
        return f

    def interpolate_fft(self, ys, omega):
        """
        Returns a polynoial f of given degree,
        such that f(omega^i) == ys[i]
        """
        n = len(ys)
        assert n & (n-1) == 0, "n must be power of two"
        assert type(omega) is self.field_type
        assert omega ** n == 1, "must be an n'th root of unity"
        assert omega ** (n //
                         2) != 1, "must be a primitive n'th root of unity"
        coeffs = [b/n for b in fft_helper(ys, 1/omega, self.field)]
        return Polynomial(coeffs, self)

    def random(self, degree, y0=None):
        coeffs = [self.field.random() for _ in range(degree+1)]
        if y0 is not None:
            if type(y0) is int:
                y0 = self.field(y0)
            assert type(y0) is self.field_type
            coeffs[0] = y0
        return Polynomial(coeffs, self)

    def interp_extrap(self, xs, omega):
        """
        Interpolates the polynomial based on the even points omega^2i
        then evaluates at all points omega^i
        """
        n = len(xs)
        assert n & (n-1) == 0, "n must be power of 2"
        assert pow(omega, 2*n) == 1, "omega must be 2n'th root of unity"
        assert pow(
            omega, n) != 1, "omega must be primitive 2n'th root of unity"

        # Interpolate the polynomial up to degree n
        poly = self.interpolate_fft(xs, omega**2)

        # Evaluate the polynomial
        xs2 = poly.evaluate_fft(omega, 2*n)

        return xs2

    def interp_extrap_cpp(self, xs, omega):
        """
        Interpolates the polynomial based on the even points omega^2i
        then evaluates at all points omega^i using C++ FFT routines.
        """
        n = len(xs)
        assert n & (n-1) == 0, "n must be power of 2"
        assert pow(omega, 2*n) == 1, "omega must be 2n'th root of unity"
        assert pow(omega, n) != 1, "omega must be primitive 2n'th root of unity"
        p = omega.modulus

        # Interpolate the polynomial up to degree n
        poly = fft_interpolate_cpp(list(range(n)), xs, (pow(omega, 2)).value, p, n)

        # Evaluate the polynomial
        xs2 = fft_cpp(poly, omega.value, p, 2*n)

        return xs2


class Polynomial(object):
    def __init__(self, coeffs, poly_cache):
        self.coeffs = list(strip_trailing_zeros(coeffs))
        self.poly_cache = poly_cache
        self.field = poly_cache.field
        for i in range(len(self.coeffs)):
            if type(self.coeffs[i]) is int:
                self.coeffs[i] = self.field(self.coeffs[i])
            assert type(self.coeffs[i]) is self.poly_cache.field_type

    def is_zero(self):
        return self.coeffs == [] or (len(self.coeffs) == 1 and self.coeffs[0] == 0)

    def __repr__(self):
        if self.is_zero():
            return '0'
        return ' + '.join(['%s x^%d' % (a, i) if i > 0 else '%s' % a
                           for i, a in enumerate(self.coeffs)])

    def __call__(self, x):
        y = self.field(0)
        xx = self.field(1)
        for coeff in self.coeffs:
            y += coeff * xx
            xx *= x
        return y

    def evaluate_fft(self, omega, n):
        assert n & (n-1) == 0, "n must be power of two"
        assert type(omega) is self.poly_cache.field_type
        assert omega ** n == 1, "must be an n'th root of unity"
        assert omega ** (n //
                         2) != 1, "must be a primitive n'th root of unity"
        return fft(self, omega, n)

    # the valuation only gives 0 to the zero polynomial, i.e. 1+degree
    def __abs__(self): return len(self.coeffs)

    def __iter__(self): return iter(self.coeffs)

    def __sub__(self, other): return self + (-other)

    def __neg__(self): return Polynomial([-a for a in self], self.poly_cache)

    def __len__(self): return len(self.coeffs)

    def __add__(self, other):
        new_coefficients = [sum(x) for x in zip_longest(
            self, other, fillvalue=self.field(0))]
        return Polynomial(new_coefficients, self.poly_cache)

    def __mul__(self, other):
        if self.is_zero() or other.is_zero():
            return self.zero()

        new_coeffs = [self.field(0)
                      for _ in range(len(self) + len(other) - 1)]

        for i, a in enumerate(self):
            for j, b in enumerate(other):
                new_coeffs[i+j] += a*b
        return Polynomial(new_coeffs, self.poly_cache)

    def degree(self): return abs(self) - 1

    def leading_coefficient(self): return self.coeffs[-1]

    def __divmod__(self, divisor):
        quotient, remainder = self.zero(), self
        divisor_deg = divisor.degree()
        divisor_lc = divisor.leading_coefficient()

        while remainder.degree() >= divisor_deg:
            monomial_exponent = remainder.degree() - divisor_deg
            monomial_zeros = [self.field(0)
                              for _ in range(monomial_exponent)]
            monomial_divisor = Polynomial(
                monomial_zeros + [remainder.leading_coefficient() / divisor_lc],
                self.poly_cache)

            quotient += monomial_divisor
            remainder -= monomial_divisor * divisor

        return quotient, remainder

    def __truediv__(self, divisor):
        if divisor.isZero():
            raise ZeroDivisionError
        return divmod(self, divisor)[0]

    def __mod__(self, divisor):
        if divisor.isZero():
            raise ZeroDivisionError
        return divmod(self, divisor)[1]

    def zero(self):
        return Polynomial([], self.poly_cache)


def polynomials_over(field):
    return PolyCache(field)


def get_omega(field, n, seed=None):
    """
    Given a field, this method returns an n^th root of unity.
    If the seed is not None then this method will return the
    same n'th root of unity for every run with the same seed

    This only makes sense if n is a power of 2!
    """
    assert n & n-1 == 0, "n must be a power of 2"
    x = field.random(seed)
    y = pow(x, (field.modulus-1)//n)
    if y == 1 or pow(y, n//2) == 1:
        return get_omega(field, n)
    assert pow(y, n) == 1, "omega must be 2n'th root of unity"
    assert pow(y, n//2) != 1, "omega must be primitive 2n'th root of unity"
    return y


def fft_helper(a, omega, field):
    """
    Given coefficients A of polynomial this method does FFT and returns
    the evaluation of the polynomial at [omega^0, omega^(n-1)]

    If the polynomial is a0*x^0 + a1*x^1 + ... + an*x^n then the coefficients
    list is of the form [a0, a1, ... , an].
    """
    n = len(a)
    assert not (n & (n-1)), "n must be a power of 2"

    if n == 1:
        return a

    b, c = a[0::2], a[1::2]
    b_bar = fft_helper(b, pow(omega, 2), field)
    c_bar = fft_helper(c, pow(omega, 2), field)
    a_bar = [field(1)]*(n)
    for j in range(n):
        k = (j % (n//2))
        a_bar[j] = b_bar[k] + pow(omega, j) * c_bar[k]
    return a_bar


def fft(poly, omega, n, seed=None):
    assert n & n-1 == 0, "n must be a power of 2"
    assert len(poly.coeffs) <= n
    assert pow(omega, n) == 1
    assert pow(omega, n//2) != 1

    padded_coeffs = poly.coeffs + ([poly.field(0)] * (n-len(poly.coeffs)))
    return fft_helper(padded_coeffs, omega, poly.field)


def fnt_decode_step1(poly, zs, omega2, n):
    """
    This needs to be run once for decoding a batch of secret shares
    It depends only on the x values (the points the polynomial is
    evaluated at, i.e. the IDs of the parties contributing shares) so
    it can be reused for multiple batches.
    Complexity: O(n^2)

    args:
        zs is a subset of [0,n)
        omega2 is a (2*n)th root of unity

    returns:
        A(X) evaluated at 1...omega2**(2n-1)
        Ai(xi) for each xi = omega**(zi)

    where:
        omega = omega2**2
        where A(X) = prod( X - xj ) for each xj
        Ai(xi) = prod( xi - xj ) for j != i
    """
    k = len(zs)
    omega = omega2**2
    xs = [omega**z for z in zs]

    # Compute A(X)
    a_ = poly([1])
    for i in range(k):
        a_ *= poly([-xs[i], 1])
    as_ = [a_(omega2**i) for i in range(2*n)]

    # Compute all Ai(Xi)
    ais_ = []
    for i in range(k):
        ai = a_.field(1)
        for j in range(k):
            if i != j:
                ai *= xs[i] - xs[j]
        ais_.append(ai)
    return as_, ais_


def fnt_decode_step2(poly, zs, ys, as_, ais_, omega2, n):
    """
    Returns a polynomial P such that P(omega**zi) = yi

    Complexity: O(n log n)

    args:
        zs is a subset of [0,n)
        As, Ais = fnt_decode_step1(zs, omega2, n)
        omega2 is a (2*n)th root of unity

    returns:
        P  Poly
    """
    k = len(ys)
    assert len(ys) == len(ais_)
    assert len(as_) == 2 * n
    omega = omega2 ** 2

    # Compute N'(x)
    nis = [ys[i] / ais_[i] for i in range(k)]
    ncoeffs = [0 for _ in range(n)]
    for i in range(k):
        ncoeffs[zs[i]] = nis[i]
    n_ = poly(ncoeffs)

    # Compute P/A(X)
    nevals = n_.evaluate_fft(omega, n)
    power_a = -poly(nevals[::-1])
    pas = power_a.evaluate_fft(omega2, 2*n)

    # Recover P(X)
    ps = [p * a for (p, a) in zip(pas, as_)]
    prec = poly.interpolate_fft(ps, omega2)
    prec.coeffs = prec.coeffs[:k]
    return prec


class EvalPoint(object):
    """Helper to generate evaluation points for polynomials between n parties

    If FFT is being used:
    omega is a root of unity s.t. order(omega) = (smallest power of 2 >= n)
    i'th point (zero-indexed) = omega^(i)

    Without FFT:
    i'th point (zero-indexed) = i + 1
    """

    def __init__(self, field, n, use_fft=False):
        self.use_fft = use_fft
        self.field = field
        self.n = n
        # Need an additional point where we evaluate polynomial to get secret
        order = n
        if use_fft:
            self.order = order if (order & (order - 1) == 0) else 2 ** order.bit_length()

            # All parties must use the same omega for FFT to work with batch
            # reconstruction. Fixing the seed to 0 is a simple way to do this.
            self.omega2 = get_omega(field, 2 * self.order, seed=0)
            self.omega = self.omega2 ** 2
        else:
            self.order = order
            self.omega2 = None
            self.omega = None

    def __call__(self, i):
        if self.use_fft:
            return self.field(self.omega2.value ** (2 * i))
        else:
            return self.field(i + 1)

    def zero(self):
        return self.field(0)


if __name__ == "__main__":
    field = GF(Subgroup.BLS12_381)
    Poly = polynomials_over(field)
    poly = Poly.random(degree=7)
    poly = Poly([1, 5, 3, 15, 0, 3])
    n = 2**3
    omega = get_omega(field, n, seed=1)
    omega2 = get_omega(field, n, seed=4)
    # FFT
    # x = fft(poly, omega=omega, n=n, test=True, enable_profiling=True)
    x = poly.evaluate_fft(omega, n)
    # IFFT
    x2 = [b/n for b in fft_helper(x, 1/omega, field)]
    poly2 = Poly.interpolate_fft(x2, omega)
    logging.info(poly2)

    logging.info(f'omega1: {omega ** (n//2)}')
    logging.info(f'omega2: {omega2 ** (n//2)}')

    logging.info('eval:')
    omega = get_omega(field, 2*n)
    for i in range(len(x)):
        logging.info(f'{omega**(2*i)} {x[i]}')
    logging.info('interp_extrap:')
    x3 = Poly.interp_extrap(x, omega)
    for i in range(len(x3)):
        logging.info(f'{omega**i} {x3[i]}')

    logging.info("How many omegas are there?")
    for i in range(10):
        omega = get_omega(field, 2**20)
        logging.info(f'{omega} {omega**(2**17)}')
