
INLINE Colorf::Colorf() : r(0), g(0), b(0), a(0)
{
}
INLINE Colorf::Colorf(float s) : r(s), g(s), b(s), a(s)
{
}
INLINE Colorf::Colorf(float r, float g, float b, float a) : r(r), g(g), b(b), a(a)
{
}

INLINE Colorf Colorf::operator+(const Colorf& v) const
{
    return Colorf(r + v.r, g + v.g, b + v.b, a + v.a);
}
INLINE Colorf Colorf::operator-(const Colorf& v) const
{
    return Colorf(r - v.r, g - v.g, b - v.b, a - v.a);
}
INLINE Colorf Colorf::operator*(float d) const
{
    return Colorf(r * d, g * d, b * d, a * d);
}
INLINE Colorf Colorf::operator/(float d) const
{
    return Colorf(r / d, g / d, b / d, a / d);
}

INLINE Colorf Colorf::operator+=(const Colorf& v)
{
    r += v.r;
    g += v.g;
    b += v.b;
    a += v.a;
    return *this;
}
INLINE Colorf Colorf::operator*=(float d)
{
    r *= d;
    g *= d;
    b *= d;
    a *= d;
    return *this;
}
INLINE Colorf Colorf::operator/=(float d)
{
    r /= d;
    g /= d;
    b /= d;
    a /= d;
    return *this;
}

INLINE Colorf operator*(float d, const Colorf v)
{
    return v.operator*(d);
}

INLINE Colorf Colorf::operator*(double d) const
{
    return operator*(float(d));
}
INLINE Colorf Colorf::operator/(double d) const
{
    return operator/(float(d));
}

INLINE Colorf Colorf::operator*=(double d)
{
    return operator*=(float(d));
}
INLINE Colorf Colorf::operator/=(double d)
{
    return operator/=(float(d));
}

INLINE Colorf operator*(double d, const Colorf v)
{
    return v.operator*(d);
}
