# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPUStableFluids is a Unity implementation of Jos Stam's Stable Fluids algorithm with GPU compute shaders. It provides both 2D and 3D fluid simulations with atmospheric physics (cloud condensation, precipitation, buoyancy).

**Requirements**: Unity 2022.3 or later

## Architecture

### Simulation Pipeline (per frame)
1. **UpdateVelocity**: Advect velocity → Apply forces (vorticity, buoyancy) → Pressure projection → Boundary conditions
2. **UpdateDensity**: Advect density → Update thermodynamics (condensation, precipitation, phase changes) → Boundary conditions
3. **UpdateDisplay**: Convert to renderable format

### Key Files
- `Assets/Scripts/Stable Fluids/StableFluids2D.cs` - 2D simulation controller
- `Assets/Scripts/Stable Fluids/StableFluids3D.cs` - 3D simulation controller with volume rendering
- `Assets/Compute/StableFluids2D.compute` - 2D GPU kernels (8x8x1 thread groups)
- `Assets/Compute/StableFluids3D.compute` - 3D GPU kernels (4x4x4 thread groups)
- `Assets/Compute/Thermodynamics.hlsl` - Shared atmospheric physics (constants, saturation, Exner function)
- `Assets/Shaders/Volume.shader` - Ray-marching volume renderer for 3D visualization

### Numerical Methods
- **Semi-Lagrangian Advection**: Backtrace along velocity with bilinear/trilinear interpolation
- **Pressure Projection**: Poisson solver via Jacobi iteration (5 iterations 2D, 10 iterations 3D) to enforce incompressibility
- **Vorticity Confinement**: Curl-based force to preserve rotational motion

### GPU Buffer Pattern
All simulations use double-buffered RenderTextures (in/out pairs) to avoid read-write conflicts. Textures are allocated in `Start()`, managed during simulation, and released on destroy.

## Important Platform Notes

**Windows Compatibility**: Cannot bind the same texture to multiple shader variables. The codebase copies velocity to `_velocityTempTexture` before advection (see StableFluids2D.cs:234-236, StableFluids3D.cs:277-279). Any new kernels that read and write the same field must follow this pattern.

## Scenes
- `Assets/Scenes/2D Stable Fluids.unity` - 2D demo (left-click stir, right-click add dye, R reset)
- `Assets/Scenes/3D Stable Fluids.unity` - 3D demo with orbit camera (left-click rotate, right-click/scroll zoom, R reset)

## Render Texture Formats
| Field | 2D | 3D |
|-------|----|----|
| Density | RGBFloat (3-channel) | ARGBFloat (4-channel) |
| Velocity | RGFloat (2-channel) | ARGBFloat (3-channel) |
| Pressure | RFloat (1-channel) | RFloat (1-channel) |
| Atmosphere LUT | RGFloat (temp + pressure) | RGFloat (temp + pressure) |

## Thermodynamics System
The density texture stores thermodynamic state:
- `.x` = water vapor mixing ratio (q_v)
- `.y` = cloud water content (q_c)
- `.z` = potential temperature perturbation (θ')

Physics constants in `Thermodynamics.hlsl`: gravity (G=9.81), gas constant (R_D=287), specific heat (C_P=1003.5), latent heat (L=2501000).
