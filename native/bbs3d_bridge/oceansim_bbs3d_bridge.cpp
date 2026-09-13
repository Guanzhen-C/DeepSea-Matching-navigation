#include <gpu_bbs3d/bbs3d.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

void copy_error(const std::string& message, char* destination,
                std::size_t destination_size) {
  if (destination == nullptr || destination_size == 0) {
    return;
  }
  const std::size_t count = std::min(message.size(), destination_size - 1);
  std::memcpy(destination, message.data(), count);
  destination[count] = '\0';
}

std::vector<Eigen::Vector3f> make_points(const float* values,
                                         std::size_t count) {
  std::vector<Eigen::Vector3f> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    points.emplace_back(values[3 * index], values[3 * index + 1],
                        values[3 * index + 2]);
  }
  return points;
}

Eigen::Vector3f make_vector(const float* values) {
  return Eigen::Vector3f(values[0], values[1], values[2]);
}

}  // namespace

extern "C" int oceansim_bbs3d_localize(
    const float* target_xyz, std::size_t target_count, const float* source_xyz,
    std::size_t source_count, float minimum_level_resolution, int maximum_level,
    const float* minimum_translation_xyz, const float* maximum_translation_xyz,
    const float* minimum_rpy, const float* maximum_rpy,
    float score_threshold_fraction, int timeout_milliseconds,
    float* output_transform_row_major, int* output_score,
    float* output_score_fraction, double* output_elapsed_milliseconds,
    int* output_localized, int* output_timed_out, char* error_message,
    std::size_t error_message_size) {
  try {
    if (target_xyz == nullptr || source_xyz == nullptr ||
        minimum_translation_xyz == nullptr || maximum_translation_xyz == nullptr ||
        minimum_rpy == nullptr || maximum_rpy == nullptr ||
        output_transform_row_major == nullptr || output_score == nullptr ||
        output_score_fraction == nullptr || output_elapsed_milliseconds == nullptr ||
        output_localized == nullptr || output_timed_out == nullptr) {
      copy_error("null pointer passed to 3D-BBS bridge", error_message,
                 error_message_size);
      return 1;
    }
    if (target_count == 0 || source_count == 0) {
      copy_error("3D-BBS target and source must be non-empty", error_message,
                 error_message_size);
      return 2;
    }
    if (!(minimum_level_resolution > 0.0f) || maximum_level < 0 ||
        score_threshold_fraction < 0.0f || score_threshold_fraction >= 1.0f) {
      copy_error("invalid 3D-BBS scalar configuration", error_message,
                 error_message_size);
      return 3;
    }

    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        output_transform_row_major[4 * row + column] =
            row == column ? 1.0f : 0.0f;
      }
    }

    const auto target = make_points(target_xyz, target_count);
    const auto source = make_points(source_xyz, source_count);
    auto solver = std::make_unique<gpu::BBS3D>();
    solver->set_tar_points(target, minimum_level_resolution, maximum_level);
    solver->set_trans_search_range(make_vector(minimum_translation_xyz),
                                   make_vector(maximum_translation_xyz));
    solver->set_angular_search_range(make_vector(minimum_rpy),
                                     make_vector(maximum_rpy));
    solver->set_score_threshold_percentage(score_threshold_fraction);
    if (timeout_milliseconds > 0) {
      solver->enable_timeout();
      solver->set_timeout_duration_in_msec(timeout_milliseconds);
    } else {
      solver->disable_timeout();
    }
    solver->set_src_points(source);
    solver->localize();

    if (solver->has_localized()) {
      const Eigen::Matrix4f transform = solver->get_global_pose();
      for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
          output_transform_row_major[4 * row + column] = transform(row, column);
        }
      }
    }
    *output_score = solver->get_best_score();
    *output_score_fraction = solver->get_best_score_percentage();
    *output_elapsed_milliseconds = solver->get_elapsed_time();
    *output_localized = solver->has_localized() ? 1 : 0;
    *output_timed_out = solver->has_timed_out() ? 1 : 0;
    copy_error("", error_message, error_message_size);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error_message, error_message_size);
    return 4;
  } catch (...) {
    copy_error("unknown exception in 3D-BBS bridge", error_message,
               error_message_size);
    return 5;
  }
}
