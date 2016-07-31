
template<typename TImage>
EnergyFunction<TImage>::EnergyFunction(double alpha) :
_alpha(alpha),
_pair_count(0)
{
}

template<typename TImage>
INLINE double EnergyFunction<TImage>::unary_term(const Vec3i& p, const Vec3d& def)
{
    double dataterm = 0;
    for (long i = 0; i < _pair_count; i++)
    {
        dataterm += pow(abs(_fixed_image[i](p) -
            _moving_image[i].linear_at(Vec3d(p) + def)), 2);
    }

    return (1 - _alpha)*dataterm;
}
template<typename TImage>
INLINE double EnergyFunction<TImage>::binary_term(const Vec3d& def1, const Vec3d& def2, const Vec3i& step)
{
    Vec3d diff = (def1 - def2) * _moving_spacing;

    double n = (step * _fixed_spacing).length_squared();
    return _alpha*diff.length_squared() / n; // TODO: Spacing?
}
template<typename TImage>
void EnergyFunction<TImage>::set_images(const Image* fixed_image, const Image* moving_image, int pair_count)
{
    assert(fixed_image && moving_image);
    assert(pair_count > 0);

    _pair_count = pair_count;
    _fixed_image.resize(_pair_count);
    _moving_image.resize(_pair_count);

    for (int i = 0; i < _pair_count; ++i)
    {
        _fixed_image[i] = fixed_image[i];
        _moving_image[i] = moving_image[i];
    }

    _fixed_spacing = _fixed_image[0].spacing();
    _moving_spacing = _moving_image[0].spacing();
}

template<>
INLINE double EnergyFunction<ImageColorf>::unary_term(const Vec3i& , const Vec3d& )
{
    assert(false);
    return 0;
    //assert(_fixed_image && _moving_image);
    //Colorf diff = _fixed_image(p) - _moving_image.linear_at(Vec3d(p) + def);
    //double dataterm = diff.r*diff.r + diff.g*diff.g + diff.b*diff.b + diff.a*diff.a;

    //return (1 - _alpha)*dataterm;
}
