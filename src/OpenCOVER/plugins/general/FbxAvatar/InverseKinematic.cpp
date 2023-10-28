#include "InverseKinematic.h"
#include <cmath>
#include <stdint.h>
#include <algorithm>
#include <iostream>
using namespace osg;

IKTwoBoneJob::IKTwoBoneJob()
  : mid_axis(osg::Z_AXIS)
  , pole_vector(osg::X_AXIS)
 {}

bool IKTwoBoneJob::Validate() const {
  bool valid = true;
  valid &= mid_axis.length2() == 1;
  return valid;
}

namespace {

// Local data structure used to share constant data accross ik stages.
struct IKConstantSetup {
  IKConstantSetup(const IKTwoBoneJob& _job) {
    
    // Computes inverse matrices required to change to start and mid spaces.
    // If matrices aren't invertible, they'll be all 0 (ozz::math
    // implementation), which will result in identity correction quaternions.
    osg::Matrix inv_start_joint, inv_mid_joint;
    inv_start_joint.invert(_job.start_joint);
    inv_mid_joint.invert(_job.mid_joint);

    // Transform some positions to mid joint space (_ms)
    const auto start_ms = inv_mid_joint *  _job.start_joint.getTrans();
    const auto end_ms = inv_mid_joint * _job.end_joint.getTrans();

    // Transform some positions to start joint space (_ss)
    const Vec3f mid_ss = inv_start_joint * _job.mid_joint.getTrans();
    const Vec3f end_ss = inv_start_joint * _job.end_joint.getTrans();

    // Computes bones vectors and length in mid and start spaces.
    // Start joint position will be treated as 0 because all joints are
    // expressed in start joint space.
    start_mid_ms = -start_ms;
    mid_end_ms = end_ms;
    start_mid_ss = mid_ss;
    const Vec3f mid_end_ss = end_ss - mid_ss;
    const Vec3f start_end_ss = end_ss;
    start_mid_ss_len2 = start_mid_ss.length2();
    mid_end_ss_len2 = mid_end_ss.length2();
    start_end_ss_len2 = start_end_ss.length2();
  }

  // Inverse matrices
  Matrix inv_start_joint;

  // Bones vectors and length in mid and start spaces (_ms and _ss).
  Vec3f start_mid_ms;
  Vec3f mid_end_ms;
  Vec3f start_mid_ss;
  float start_mid_ss_len2;
  float mid_end_ss_len2;
  float start_end_ss_len2;
};

// Smoothen target position when it's further that a ratio of the joint chain
// length, and start to target length isn't 0.
// Inspired by http://www.softimageblog.com/archives/108
// and http://www.ryanjuckett.com/programming/analytic-two-bone-ik-in-2d/
bool SoftenTarget(const IKTwoBoneJob& _job, const IKConstantSetup& _setup,
                  Vec3f* _start_target_ss,
                  Vec3f* _start_target_ss_len2) {
  // Hanlde position in start joint space (_ss)
  const Vec3f start_target_original_ss = _setup.inv_start_joint * _job.target;
  const float start_target_original_ss_len2 = start_target_original_ss.length2();

  const float start_mid_ss_len = _setup.start_mid_ss_len2;
  const float mid_end_ss_len =  _setup.mid_end_ss_len2;
  const float start_target_original_ss_len = start_target_original_ss_len2;
  float bone_len_diff_abs = std::abs(start_mid_ss_len - mid_end_ss_len);
  const auto bones_chain_len = start_mid_ss_len + mid_end_ss_len;
  auto da =  _job.soften;
  da = std::clamp(da, 0.0f, 1.0f);
  const auto ds = bones_chain_len - da;

  // Sotftens target position if it is further than a ratio (_soften) of the
  // whole bone chain length. Needs to check also that ds and
  // start_target_original_ss_len2 are != 0, because they're used as a
  // denominator.
  // x = start_target_original_ss_len > da
  // y = start_target_original_ss_len > 0
  // z = start_target_original_ss_len > bone_len_diff_abs
  // w = ds                           > 0

  if(start_target_original_ss_len > da && start_target_original_ss_len > 0 && start_target_original_ss_len > bone_len_diff_abs && ds > 0)
  {
    // Finds interpolation ratio (aka alpha).
    const float alpha = (start_target_original_ss_len - da) / ds;
    // Approximate an exponential function with : 1-(3^4)/(alpha+3)^4
    // The derivative must be 1 for x = 0, and y must never exceeds 1.
    // Negative x aren't used.

    const auto factor = std::pow(alpha + 3, 4);
    const Vec3f ratio = Vec3f{81 / factor, 1, 81 / factor};
    
    // Recomputes start_target_ss vector and length.
    const Vec3f start_target_ss_len; //= da + ds - ds * ratio;
    for (size_t i = 0; i < 3; i++)
    {
      auto factor = da + ds - ds * ratio[i];
      (*_start_target_ss_len2)[i] = factor * factor;
      (*_start_target_ss)[i] = start_target_original_ss[i] * start_target_ss_len.x() / start_target_original_ss_len;
    }
  } else {
    *_start_target_ss = start_target_original_ss;
    *_start_target_ss_len2 = osg::Vec3f(start_target_original_ss_len2, start_target_original_ss_len2, start_target_original_ss_len2);
  }

  // The maximum distance we can reach is the soften bone chain length: da
  // (stored in !x). The minimum distance we can reach is the absolute value of
  // the difference of the 2 bone lengths, |d1−d2| (stored in z). x is 0 and z
  // is 1, yw are untested.
  return start_target_original_ss_len == 0 && start_target_original_ss_len == 1;
}

Quat ComputeMidJoint(const IKTwoBoneJob& _job,
                               const IKConstantSetup& _setup,
                               Vec3f _start_target_ss_len2) {
  // Computes expected angle at mid_ss joint, using law of cosine (generalized
  // Pythagorean).
  // c^2 = a^2 + b^2 - 2ab cosC
  // cosC = (a^2 + b^2 - c^2) / 2ab
  // Computes both corrected and initial mid joint angles
  // cosine within a single Vec3f (corrected is x component, initial is y).
  const float start_mid_end_sum_ss_len2 =
      _setup.start_mid_ss_len2 + _setup.mid_end_ss_len2;
  const float start_mid_end_ss_half_rlen = 0.5f / sqrtf(_setup.start_mid_ss_len2 * _setup.mid_end_ss_len2);

  // Cos value needs to be clamped, as it will exit expected range if
  // start_target_ss_len2 is longer than the triangle can be (start_mid_ss +
  // mid_end_ss).
  Vec3f mid_cos_angles;
  auto _start_target_ss_len2_mod = _start_target_ss_len2;
  _start_target_ss_len2_mod.y() = _start_target_ss_len2.x();
  for (size_t i = 0; i < 3; i++)
  {
    mid_cos_angles[i] = start_mid_end_sum_ss_len2 - _start_target_ss_len2_mod[i] * start_mid_end_ss_half_rlen;
    mid_cos_angles[i] = std::clamp(mid_cos_angles[i], -1.0f, 1.0f);
  }

  // Computes corrected angle
  const float mid_corrected_angle = acos(mid_cos_angles.x());

  // Computes initial angle.
  // The sign of this angle needs to be decided. It's considered negative if
  // mid-to-end joint is bent backward (mid_axis direction dictates valid
  // bent direction).
  const Vec3f bent_side_ref =  _setup.start_mid_ms ^ _job.mid_axis;
  const int bent_side_flip = bent_side_ref * _setup.mid_end_ms < 0 ? -1 : 1;

  const float mid_initial_angle = acos(mid_cos_angles.y()) * bent_side_flip;

  // Finally deduces initial to corrected angle difference.
  const float mid_angles_diff = mid_corrected_angle - mid_initial_angle;

  // Builds queternion.
  return Quat(mid_angles_diff, _job.mid_axis);
}

Quat ComputeStartJoint(const IKTwoBoneJob& _job,
                                 const IKConstantSetup& _setup,
                                 const Quat& _mid_rot_ms,
                                 Vec3f _start_target_ss,
                                 Vec3f _start_target_ss_len) {
  // Pole vector in start joint space (_ss)
  const Vec3f pole_ss = _setup.inv_start_joint * _job.pole_vector;

  // start_mid_ss with quaternion mid_rot_ms applied.
  auto m1 = _mid_rot_ms * _setup.mid_end_ms;
  auto m2 = _job.mid_joint * m1;
  const Vec3f mid_end_ss_final = _setup.inv_start_joint * m2;
  const Vec3f start_end_ss_final = _setup.start_mid_ss + mid_end_ss_final;

  // Quaternion for rotating the effector onto the target
  Quat end_to_target_rot_ss;
  end_to_target_rot_ss.makeRotate(start_end_ss_final, _start_target_ss);

  // Calculates rotate_plane_ss quaternion which aligns joint chain plane to
  // the reference plane (pole vector). This can only be computed if start
  // target axis is valid (not 0 length)
  // -------------------------------------------------
  Quat start_rot_ss = end_to_target_rot_ss;
  if (_start_target_ss_len.x() > 0 && _start_target_ss_len.y() > 0 && _start_target_ss_len.z() > 0) {
    // Computes each plane normal.
    Vec3f ref_plane_normal_ss = _start_target_ss ^ pole_ss;
    float ref_plane_normal_ss_len = ref_plane_normal_ss.length();
    // Computes joint chain plane normal, which is the same as mid joint axis
    // (same triangle).
    const Vec3f mid_axis_ss = _setup.inv_start_joint * (_job.mid_joint * _job.mid_axis);
    Vec3f joint_plane_normal_ss = end_to_target_rot_ss * mid_axis_ss;
    const float joint_plane_normal_ss_len = joint_plane_normal_ss.length();
    
    const Vec3f rsqrts(1/_start_target_ss_len.x(), 1/ref_plane_normal_ss_len, 1/joint_plane_normal_ss_len);
    // Computes angle cosine between the 2 normalized normals.
    for (size_t i = 0; i < 3; i++)
    {
      ref_plane_normal_ss[i] = ref_plane_normal_ss[i] / ref_plane_normal_ss_len;
      joint_plane_normal_ss[i] = joint_plane_normal_ss[i] / joint_plane_normal_ss_len;
    }
    

    const float rotate_plane_cos_angle = ref_plane_normal_ss * joint_plane_normal_ss;

    // Computes rotation axis, which is either start_target_ss or
    // -start_target_ss depending on rotation direction.
    const int start_axis_flip = joint_plane_normal_ss * pole_ss < 0? -1 : 1;
    Vec3f rotate_plane_axis_ss = _start_target_ss;
    for (size_t i = 0; i < 3; i++)
    {
      rotate_plane_axis_ss[i] *= 1/_start_target_ss_len.x(); //maybe use sqrt(_start_target_ss_len)
    }

    const Vec3f rotate_plane_axis_flipped_ss = rotate_plane_axis_ss;
    if(start_axis_flip == -1)
    {
      for (size_t i = 0; i < 3; i++)
      {
        rotate_plane_axis_ss[i] = std::abs(rotate_plane_axis_ss[i]) * -1;
      }
    }

    // Builds quaternion along rotation axis.
    const Quat rotate_plane_ss(cos(std::clamp(rotate_plane_cos_angle, -1.0f, 1.0f)),  rotate_plane_axis_flipped_ss);

    if (_job.twist_angle != 0.f) {
      // If a twist angle is provided, rotation angle is rotated along
      // rotation plane axis.
      const Quat twist_ss(_job.twist_angle, rotate_plane_axis_ss);

      start_rot_ss = twist_ss * rotate_plane_ss * end_to_target_rot_ss;
    } else {
      start_rot_ss = rotate_plane_ss * end_to_target_rot_ss;
    }
  }
  return start_rot_ss;
}

void WeightOutput(IKTwoBoneJob& _job, const IKConstantSetup& _setup,
                  const Quat& _start_rot,
                  const Quat& _mid_rot) {

  // Fix up quaternions so w is always positive, which is required for NLerp
  // (with identity quaternion) to lerp the shortest path.
  auto start_rot_fu = _start_rot;
  start_rot_fu.w() = std::abs(start_rot_fu.w());

  auto mid_rot_fu = _mid_rot;
  mid_rot_fu.w() = std::abs(mid_rot_fu.w());

  if (_job.weight < 1.f) {
    // NLerp start and mid joint rotations.
    const Quat identity(0,0,0,1);
    const float simd_weight = std::max(0.0f, _job.weight);

    // Lerp
    Quat start_lerp;
    Quat mid_lerp;
  
    for (size_t i = 0; i < 4; i++)
    {
      start_lerp[i] = identity[i] + simd_weight * (start_rot_fu[i] -  identity[i]);
      mid_lerp[i] = identity[i] + simd_weight * (mid_rot_fu[i] -  identity[i]);
    }
    // Normalize
    for (size_t i = 0; i < 4; i++)
    {
      start_lerp[i] = start_lerp[i] / start_lerp.length();
      mid_lerp[i] = mid_lerp[i] / start_lerp.length();
    }
    _job.start_joint_correction = start_lerp;
    _job.mid_joint_correction = mid_lerp;
  } else {
    // Quatenions don't need interpolation
    _job.start_joint_correction = start_rot_fu;
    _job.mid_joint_correction = mid_rot_fu;
  }
}
}  // namespace

bool IKTwoBoneJob::Run() {
  if (!Validate()) {
    return false;
  }

  // Early out if weight is 0.
  if (weight <= 0.f) {
    // No correction.
    start_joint_correction = mid_joint_correction = Quat();
    // Target isn't reached.
    if (reached) {
      *reached = false;
    }
    return true;
  }

  // Prepares constant ik data.
  const IKConstantSetup setup(*this);

  // Finds soften target position.
  Vec3f start_target_ss;
  Vec3f start_target_ss_len2;
  const bool lreached = SoftenTarget(*this, setup, &start_target_ss, &start_target_ss_len2);
  if (reached) {
    *reached = lreached && weight >= 1.f;
  }
  std::cerr << "start_target_ss " << start_target_ss.x() << ", " << start_target_ss.y() << ", " << start_target_ss.z() << std::endl;

  // Calculate mid_rot_local quaternion which solves for the mid_ss joint
  // rotation.
  const Quat mid_rot_ms =
      ComputeMidJoint(*this, setup, start_target_ss_len2);

  // Calculates end_to_target_rot_ss quaternion which solves for effector
  // rotating onto the target.
  const Quat start_rot_ss = ComputeStartJoint(
      *this, setup, mid_rot_ms, start_target_ss, start_target_ss_len2);

  // Finally apply weight and output quaternions.
  WeightOutput(*this, setup, start_rot_ss, mid_rot_ms);

  return true;
}
