#ifndef __CORE_IMAGE_COLOR_H__
#define __CORE_IMAGE_COLOR_H__

struct RGBA32
{
    uint8_t r, g, b, a;
};

struct Colorf
{
    float r, g, b, a;

    Colorf();
    explicit Colorf(float s);
    explicit Colorf(float r, float g, float b, float a);

    Colorf operator+(const Colorf& v) const;
    Colorf operator-(const Colorf& v) const;
    Colorf operator*(float d) const;
    Colorf operator/(float d) const;

    Colorf operator+=(const Colorf& v);
    Colorf operator*=(float d);
    Colorf operator/=(float d);

    Colorf operator*(double d) const;
    Colorf operator/(double d) const;

    Colorf operator*=(double d);
    Colorf operator/=(double d);
};

#include "Color.inl"

#endif // __CORE_IMAGE_COLOR_H__
