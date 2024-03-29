#ifndef __REGISTRATION_BLOCKED_GRAPHCUT_OPTIMIZER_H__
#define __REGISTRATION_BLOCKED_GRAPHCUT_OPTIMIZER_H__

#include "EnergyFunction.h"
#include "Optimizer.h"

template<typename TImage>
class BlockedGraphCutOptimizer : public Optimizer
{
public:
    BlockedGraphCutOptimizer();
    ~BlockedGraphCutOptimizer();

    void execute(
        const Image* fixed, 
        const Image* moving, 
        int pair_count,
        const ImageUInt8& constraint_mask,
        const ImageVec3d& constraint_values,
        ImageVec3d& def) OVERRIDE;

private:
    bool do_block(
        const Vec3i& block_p, 
        const Vec3i& block_dims, 
        const Vec3i& block_offset, 
        const Vec3d& delta,
        const ImageUInt8& constraint_mask,
        ImageVec3d& def);

    Vec3i _neighbors[6];
    double _epsilon;

    EnergyFunction<TImage> _energy;
};

#include "BlockedGraphCutOptimizer.inl"

#endif // __REGISTRATION_BLOCKED_GRAPHCUT_OPTIMIZER_H__
