/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void block_reduction(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists));
    g.sync();
  }
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
	
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

    __shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
    __shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
    __shared__ float3 dL_dtransMat_shared_1[BLOCK_SIZE];
	__shared__ float3 dL_dtransMat_shared_2[BLOCK_SIZE];
	__shared__ float3 dL_dtransMat_shared_3[BLOCK_SIZE];	
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];

#if RENDER_AXUTILITY
    __shared__ float3 dL_dnormal3D_shared[BLOCK_SIZE];
#endif

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0.f;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0.f;

	float accum_rec[C] = { 0.f };
	float dL_dpixel[C] = { 0.f };

#if RENDER_AXUTILITY
	float dL_dreg = 0.f;
	float dL_ddepth = 0.f;
	float dL_daccum = 0.f;
	float dL_dnormal2D[3]  = { 0.f };
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0.f;
	float dL_dmedian_depth = 0.f;
	float dL_dmax_dweight = 0.f;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		#pragma unroll
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0.f;
	float last_normal[3] = { 0.f };
	float accum_depth_rec = 0.f;
	float accum_alpha_rec = 0.f;
	float accum_normal_rec[3] = {0.f};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0.f;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0.f;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0.f;
#endif

	if (inside){
		#pragma unroll
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_normal_opacity[tid] = normal_opacity[coll_id];
			collected_Tu[tid] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[tid] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[tid] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};

			#pragma unroll
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];
		}
		// block.sync();

		// Iterate over Gaussians
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{

			block.sync();
			if (tid == 0) {
				skip_counter = 0;
			}
			block.sync();

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			// contributor--;
			// if (contributor >= last_contributor)
			// 	continue;

			bool skip = done;
			contributor = done ? contributor : contributor - 1;
			skip |= contributor >= last_contributor;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			// if (p.z == 0.0) continue;
			skip |= (p.z == 0);
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			// if (c_d < near_n) continue;
			skip |= (c_d < near_n);
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			// if (power > 0.0f)
			// 	continue;

			skip |= (power > 0.0f);

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			// if (alpha < 1.0f / 255.0f)
			// 	continue;

			skip |= (alpha < 1.0f / 255.0f);


			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();

			if (skip_counter == BLOCK_SIZE) {
				continue;
			}

			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			float local_dL_dcolors[3];

			#pragma unroll
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				// atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
			}

			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

			float dL_dz = 0.0f;
			float dL_dweight = 0;


#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = skip ? last_dL_dT : dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = skip ? accum_depth_rec : last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = skip? last_depth : c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = skip ? accum_alpha_rec : last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals

			float local_dL_dnormal2D[3];
			#pragma unroll
			for (int ch = 0; ch < 3; ch++) {
				// accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				// last_normal[ch] = normal[ch];
				accum_normal_rec[ch] = skip ? accum_normal_rec[ch] : last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = skip ? last_normal[ch] : normal[ch];

				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				local_dL_dnormal2D[ch] = skip ? 0.0f : dL_dnormal2D[ch];
				// atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}

			dL_dnormal3D_shared[tid] = make_float3(local_dL_dnormal2D[0], local_dL_dnormal2D[1], local_dL_dnormal2D[2]);
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			// last_alpha = alpha;
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

				dL_dtransMat_shared_1[tid] = skip ? make_float3(0.f, 0.f, 0.f) : dL_dTu;
				dL_dtransMat_shared_2[tid] = skip ? make_float3(0.f, 0.f, 0.f) : dL_dTv;
				dL_dtransMat_shared_3[tid] = skip ? make_float3(0.f, 0.f, 0.f) : dL_dTw;
				dL_dmean2D_shared[tid] = make_float2(0.f, 0.f);

			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;


				dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx;
				dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely;

				dL_dtransMat_shared_1[tid] = make_float3(0.f, 0.f, 0.f);
				dL_dtransMat_shared_2[tid] = make_float3(0.f, 0.f, 0.f);
				dL_dtransMat_shared_3[tid] = skip ? make_float3(0.f, 0.f, 0.f) : make_float3(s.x * dL_dz, s.y * dL_dz, dL_dz);

			}

			// Update gradients w.r.t. opacity of the Gaussian
			dL_dopacity_shared[tid] = skip ? 0.f :  G * dL_dalpha;

			block_reduction(block, dL_dcolors_shared, dL_dopacity_shared, 
			dL_dtransMat_shared_1,dL_dtransMat_shared_2, dL_dtransMat_shared_3,
			dL_dmean2D_shared, dL_dnormal3D_shared);


			if (tid == 0) {
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];
				float3 dL_dnormal3D_acc = dL_dnormal3D_shared[0];
				float3 dL_dtransMat_acc_1 = dL_dtransMat_shared_1[0];
				float3 dL_dtransMat_acc_2 = dL_dtransMat_shared_2[0];
				float3 dL_dtransMat_acc_3 = dL_dtransMat_shared_3[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);

				atomicAdd(&dL_dcolors[global_id * 3 + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * 3 + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * 3 + 2], dL_dcolors_acc.z);

				atomicAdd(&dL_dnormal3D[global_id * 3 + 0], dL_dnormal3D_acc.x);
				atomicAdd(&dL_dnormal3D[global_id * 3 + 1], dL_dnormal3D_acc.y);
				atomicAdd(&dL_dnormal3D[global_id * 3 + 2], dL_dnormal3D_acc.z);

				atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dtransMat_acc_1.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dtransMat_acc_1.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dtransMat_acc_1.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dtransMat_acc_2.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dtransMat_acc_2.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dtransMat_acc_2.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dtransMat_acc_3.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dtransMat_acc_3.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dtransMat_acc_3.z);

			}

		}
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* projmatrix_raw,
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots,
	 float* dL_dtau)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	glm::mat3 L;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	glm::mat3x4 M;
	glm::mat4x3 Mt;
	glm::mat3x4 view2pix;
	glm::mat4 viewmat_glm;
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		L = R * S;

		M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);
		Mt = glm::transpose(M);

		viewmat_glm = glm::mat4(
			viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
			viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
			viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14],
			viewmatrix[3], viewmatrix[7], viewmatrix[11], viewmatrix[15]
		);

		glm::mat4 projmat_glm = glm::mat4(
			projmatrix_raw[0], projmatrix_raw[4], projmatrix_raw[8], projmatrix_raw[12],
			projmatrix_raw[1], projmatrix_raw[5], projmatrix_raw[9], projmatrix_raw[13],
			projmatrix_raw[2], projmatrix_raw[6], projmatrix_raw[10], projmatrix_raw[14],
			projmatrix_raw[3], projmatrix_raw[7], projmatrix_raw[11], projmatrix_raw[15]
		);
		

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		view2pix = projmat_glm * ndc2pix;

		

		P = viewmat_glm * view2pix;
		T = Mt * P;

		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);


	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;


	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);

	

	float dL_dT_data[9];
	dL_dT_data[0] = dL_dTs[idx*9+0]; dL_dT_data[3] = dL_dTs[idx*9+1]; dL_dT_data[6] = dL_dTs[idx*9+2];
	dL_dT_data[1] = dL_dTs[idx*9+3]; dL_dT_data[4] = dL_dTs[idx*9+4]; dL_dT_data[7] = dL_dTs[idx*9+5];
	dL_dT_data[2] = dL_dTs[idx*9+6]; dL_dT_data[5] = dL_dTs[idx*9+7]; dL_dT_data[8] = dL_dTs[idx*9+8];


	mat33 dL_dT_3x3 = mat33(
			dL_dT_data
	);
	float Mt_data[9] = {Mt[0][0], Mt[1][0], Mt[2][0],
		Mt[0][1], Mt[1][1], Mt[2][1],
		Mt[0][2], Mt[1][2], Mt[2][2]};

	mat33 Mt_3x3 = mat33(
		Mt_data
	);

	mat33 dL_dT_3x3_t = dL_dT_3x3.transpose();
	mat33 M_3x3 = Mt_3x3.transpose();

	glm::mat3x4 dL_dP_glm = M * dL_dT;
	
	float dL_dViewMat_00 = dL_dP_glm[0][0] * view2pix[0][0]; float dL_dViewMat_01 = dL_dP_glm[0][1] * view2pix[0][0]; float dL_dViewMat_02 = dL_dP_glm[0][2] * view2pix[0][0]; float dL_dViewMat_03 = dL_dP_glm[0][3] * view2pix[0][0];
	float dL_dViewMat_10 = dL_dP_glm[1][0] * view2pix[1][1]; float dL_dViewMat_11 = dL_dP_glm[1][1] * view2pix[1][1]; float dL_dViewMat_12 = dL_dP_glm[1][2] * view2pix[1][1]; float dL_dViewMat_13 = dL_dP_glm[1][3] * view2pix[1][1];
	
	float dL_dViewMat_20 = dL_dP_glm[0][0] * view2pix[0][2] + dL_dP_glm[1][0] * view2pix[1][2] + dL_dP_glm[2][0];
	float dL_dViewMat_21 = dL_dP_glm[0][1] * view2pix[0][2] + dL_dP_glm[1][1] * view2pix[1][2] + dL_dP_glm[2][1]; 
	float dL_dViewMat_22 = dL_dP_glm[0][2] * view2pix[0][2] + dL_dP_glm[1][2] * view2pix[1][2] + dL_dP_glm[2][2]; 
	float dL_dViewMat_23 = dL_dP_glm[0][3] * view2pix[0][2] + dL_dP_glm[1][3] * view2pix[1][2] + dL_dP_glm[2][3];


	dL_dViewMat_00 += dL_dnormals[idx].x *L[2].x; dL_dViewMat_01 += dL_dnormals[idx].x *L[2].y; dL_dViewMat_02 += dL_dnormals[idx].x *L[2].z;
	dL_dViewMat_10 += dL_dnormals[idx].y *L[2].x; dL_dViewMat_11 += dL_dnormals[idx].y *L[2].y; dL_dViewMat_12 += dL_dnormals[idx].y *L[2].z;
	dL_dViewMat_20 += dL_dnormals[idx].z *L[2].x; dL_dViewMat_21 += dL_dnormals[idx].z *L[2].y; dL_dViewMat_22 += dL_dnormals[idx].z *L[2].z;


	SE3 T_CW(viewmatrix);
	float3 t_ = T_CW.t();
	mat33 Rot = T_CW.R().data();

	float3 c1 = Rot.cols[0];
	float3 c2 = Rot.cols[1];
	float3 c3 = Rot.cols[2];
	mat33 skew1 = - mat33::skew_symmetric(c1);
	mat33 skew2 = - mat33::skew_symmetric(c2);
	mat33 skew3 = - mat33::skew_symmetric(c3);
	mat33 skewt = - mat33::skew_symmetric(t_);

	float dL_dView_data[12];
	dL_dView_data[0] = dL_dViewMat_00; dL_dView_data[3] = dL_dViewMat_01; dL_dView_data[6] = dL_dViewMat_02; dL_dView_data[9] = dL_dViewMat_03; 
	dL_dView_data[1] = dL_dViewMat_10; dL_dView_data[4] = dL_dViewMat_11; dL_dView_data[7] = dL_dViewMat_12; dL_dView_data[10] = dL_dViewMat_13; 
	dL_dView_data[2] = dL_dViewMat_20; dL_dView_data[5] = dL_dViewMat_21; dL_dView_data[8] = dL_dViewMat_22; dL_dView_data[11] = dL_dViewMat_23;
	mat34 dL_dView(dL_dView_data);

	float3 c1_view = dL_dView.cols[0]; float3 c2_view = dL_dView.cols[1]; 
	float3 c3_view = dL_dView.cols[2]; float3 t_view = dL_dView.cols[3];

	float3 dL_drho;
	dL_drho.x = dL_dViewMat_03; dL_drho.y = dL_dViewMat_13; dL_drho.z = dL_dViewMat_23;


	float3 dL_dtheta;
	dL_dtheta.x = dot(c1_view, skew1.cols[0]) + dot(c2_view, skew2.cols[0]) + dot(c3_view, skew3.cols[0]) + dot(t_view, skewt.cols[0]);
	dL_dtheta.y = dot(c1_view, skew1.cols[1]) + dot(c2_view, skew2.cols[1]) + dot(c3_view, skew3.cols[1]) + dot(t_view, skewt.cols[1]);
	dL_dtheta.z = dot(c1_view, skew1.cols[2]) + dot(c2_view, skew2.cols[2]) + dot(c3_view, skew3.cols[2]) + dot(t_view, skewt.cols[2]);

	float dL_dt[6] = {dL_drho.x, dL_drho.y, dL_drho.z,
	 dL_dtheta.x, dL_dtheta.y, dL_dtheta.z};

	for(int i=0; i<6; i++){
		dL_dtau[6*idx+i] += dL_dt[i];
	}

#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float* proj_raw,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float* dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, proj_raw, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots,
		dL_dtau
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
    const float focal_x, float focal_y,
	const float cx, float cy,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float* dL_dtau)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		dL_dtau
	);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors
		);
}