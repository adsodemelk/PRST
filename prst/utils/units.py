"""
Defines various units numerically in their SI units.

Units:

   Newton        - Force of one Newton, in units of Newton.
   Pascal        - Numerical value, in units of Pascal, of one Pascal.
   atm           - Numerical value, in units of Pascal, of one atmosphere.
   bar           - Numerical value, in units of Pascal, of one bar.
   centi         - One houndreth prefix.
   convertFrom   - Convert physical quantity from given unit to equivalent SI.
   convertTo     - Convert physical quantity from SI to equivalent given unit.
   darcy         - Compute numerical value, in units of m^2, of the Darcy constant.
   day           - Give numerical value, in units of seconds, of one day.
   deci          - One tenth prefix.
   dyne          - Compute numerical value, in units of Newton of one dyne.
   ft            - Distance of one foot (in units of meters).
   gallon        - Compute numerical value, in units of m^3, of one U.S. liquid gallon.
   getUnitSystem - Define unit conversion factors for input data.
   giga          - One billion (milliard) prefix.
   gram          - Mass of one gram, in units of kilogram.
   hour          - Time span of one hour (in units of seconds).
   inch          - Distance of one inch (in units of meters).
   kilo          - One thousand prefix.
   kilogram      - Mass of one kilogram, in units of kilogram.
   lbf           - Force excerted by a mass of one avoirdupois pound at Tellus equator.
   mega          - One million prefix.
   meter         - Distance of one meter (in units of meters).
   micro         - One millionth prefix.
   milli         - One thousandth prefix.
   minute        - Time span of one minute (in units of seconds).
   poise         - Compute numerical value, in units of Pa*s, of one poise (P).
   pound         - Mass of one avoirdupois pound, in units of kilogram.
   psi          - Compute numerical value, in units of Pascal, of one Psi.
   second        - Time span of one second (in units of seconds).
   stb           - Compute numerical value, in units of m^3, of one standard barrel.
   year          - Give numerical value, in units of seconds, of one year.


Functions:

    convert - Convert physical quantity between different units

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# SI prefixes
micro = 1e-6
milli = 1/1000
centi = 1/100
deci = 1/10
kilo = 1000
mega = 1e6
giga = 1e9

# SI units
meter = 1
kilogram = 1
second = 1
Newton = 1
Pascal = 1

# Derived units
inch = 2.54 * centi*meter
ft = 0.3048 * meter
darcy = 9.869232667160130e-13 * meter**2
gallon = 231 * inch**3
stb = 42 * gallon
minute = 60 * second
hour = 3600 * second
day = 24 * hour
year = 365 * day
gram = 1e-3 * kilogram
pound = 0.45359237 * kilogram
dyne = 1e-5 * Newton
lbf = 9.80665 * meter/second**2 * pound
atm = 101325 * Pascal
bar = 1e5 * Pascal
psi = lbf / inch**2
poise = 0.1 * Pascal * second

def convert(q, from_=1, to=1):
    """Convert physical quantity to different unit.

    Synopsis:
        q2 = convert(q1, from_=unit1, to=unit2)

    Arguments:
        q (ndarray or number): Physical quantity to convert from

        from_ (Optional[number]): Unit to convert from. Assumed to be SI unit
                                  by default. "from" is a reserved keyword in
                                  Python.

        to (Optional[number]): Unit to convert to. Assumed to be SI unit by
                               default.

    Example:
        >>> convert(5, from_=kilogram, to=gram)
        5000.0
        >>> convert(5, from_=kilogram, to=gram)
        5000.0
        >>> convert(5000, from_=gram, to=kilogram)
        5.0
        >>> convert(5*bar, to=bar)
        5.0

    Note: It is the caller's responsibility to supply physically consistent
    units. E.q., converting from "meter" to "hour" will not cause any error
    messages.
    """
    return q * (from_ / to)
