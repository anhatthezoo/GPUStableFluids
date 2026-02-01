using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace StableFluids
{
    /// <summary>
    /// A GPU implementation of Jos Stam's Stable Fluids in 2D.
    /// </summary>
    public class StableFluids2D : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField] private ComputeShader _stableFluids2DCompute;

        [Header("Fluid Simulation Parameters")] 
        [SerializeField] private Vector2Int _resolution;
        [SerializeField] private Vector2Int _volumeSize;
        [SerializeField] private float _diffusion;
        [SerializeField] private float _vorticity;
        [SerializeField] private float _groundTemperature;
        [SerializeField] private float _groundPressure;
        [SerializeField] private float _lapseRate;

        [Header("Testing Parameters")] 
        [SerializeField] private Texture2D _initialImage;
        [SerializeField] private float _addVelocity;
        [SerializeField] private float _addDensity;
        [SerializeField] private float _addVelocityRadius;
        [SerializeField] private float _addDensityRadius;

        // Input
        private Vector2 _previousMousePosition;

        // Render textures
        private RenderTexture _displayTexture;
        private RenderTexture _densityOutTexture;
        private RenderTexture _densityInTexture;
        private RenderTexture _velocityOutTexture;
        private RenderTexture _velocityInTexture;
        private RenderTexture _velocityTempTexture;
        private RenderTexture _pressureOutTexture;
        private RenderTexture _pressureInTexture;
        private RenderTexture _divergenceTexture;
        private RenderTexture _atmosphereLUT;
        
        // Compute kernels
        private int _updateDisplayKernel;
        private int _atmoLUTKernel;
        private int _resetThermoKernel;
        private int _addValueKernel;
        private int _advectionKernel;
        private int _applyForcesKernel;
        private int _updateThermoKernel;
        private int _projectionPt1Kernel;
        private int _projectionPt2Kernel;
        private int _projectionPt3Kernel;
        private int _setBoundsXKernel;
        private int _setBoundsYKernel;

        // Thread counts
        private Vector3Int _threadCounts;
        private int _atmoLUTThreads;
        private int _setBoundsXThreadCount;
        private int _setBoundsYThreadCount;

        private void Start()
        {
            // Get kernel ids
            _updateDisplayKernel = _stableFluids2DCompute.FindKernel("UpdateDisplay");
            _atmoLUTKernel = _stableFluids2DCompute.FindKernel("GenerateAtmoLUT");
            _resetThermoKernel = _stableFluids2DCompute.FindKernel("ResetThermodynamics");
            _addValueKernel = _stableFluids2DCompute.FindKernel("AddValue");
            _advectionKernel = _stableFluids2DCompute.FindKernel("Advection");
            _applyForcesKernel = _stableFluids2DCompute.FindKernel("ApplyForces");
            _updateThermoKernel = _stableFluids2DCompute.FindKernel("UpdateThermodynamics");
            _projectionPt1Kernel = _stableFluids2DCompute.FindKernel("ProjectionPt1");
            _projectionPt2Kernel = _stableFluids2DCompute.FindKernel("ProjectionPt2");
            _projectionPt3Kernel = _stableFluids2DCompute.FindKernel("ProjectionPt3");
            _setBoundsXKernel = _stableFluids2DCompute.FindKernel("SetBoundsX");
            _setBoundsYKernel = _stableFluids2DCompute.FindKernel("SetBoundsY");
            
            // Get thread counts
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_advectionKernel, 
                out uint xThreadGroupSize, out uint yThreadGroupSize, out uint zThreadGroupSize);
            _threadCounts = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(1 / (float)zThreadGroupSize));
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_atmoLUTKernel,
                out xThreadGroupSize, out _, out _);
            _atmoLUTThreads = Mathf.CeilToInt(_resolution.y / (float)xThreadGroupSize);
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_setBoundsXKernel, 
                out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsXThreadCount = Mathf.CeilToInt(_resolution.x * 2 / (float)xThreadGroupSize);
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_setBoundsYKernel, 
                out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsYThreadCount = Mathf.CeilToInt(_resolution.y * 2 / (float)xThreadGroupSize);

            // Pass constants to compute
            _stableFluids2DCompute.SetInts("_Resolution", new int[] { _resolution.x, _resolution.y });
            _stableFluids2DCompute.SetFloat("_VolumeHeight", _volumeSize.y);
            _stableFluids2DCompute.SetFloat("_Vorticity", _vorticity);
            _stableFluids2DCompute.SetFloat("_GroundTemperature", _groundTemperature);
            _stableFluids2DCompute.SetFloat("_GroundPressure", _groundPressure);
            _stableFluids2DCompute.SetFloat("_LapseRate", _lapseRate);

            // Subscribe to end of render pipeline event
            RenderPipelineManager.endContextRendering += OnEndContextRendering;
            
            // Allocate render textures
            _displayTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 4);
            _densityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 3);
            _densityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 3);
            _velocityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 2);
            _velocityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 2);
            _velocityTempTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 2);
            _pressureOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 1);
            _pressureInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 1);
            _divergenceTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 1);
            _atmosphereLUT = RenderTextureUtilities.AllocateRWLinearRT(_resolution.y, 1, 0, 2);

            // Reset simulation
            Reset();
        }

        /// <summary>
        /// Process input and run fluid simulation.
        /// </summary>
        void Update()
        {
            // Input handling
            Vector2 mousePosition = Input.mousePosition;
            Vector2 scaledMousePosition = mousePosition / new Vector2(Screen.width, Screen.height) * _resolution;
            if (Input.GetMouseButton(1))
            {
                Color density = new Color(0.001f, 0.0f, 2.5f, 0.0f) * Time.deltaTime;
                AddValueToTexture(_densityOutTexture, scaledMousePosition, density, _addDensityRadius);
            }
            if (Input.GetMouseButton(0))
            {
                Vector2 mouseVelocity = (mousePosition - _previousMousePosition) * _addVelocity * Time.deltaTime;
                AddValueToTexture(_velocityOutTexture, scaledMousePosition, mouseVelocity, _addVelocityRadius);
            }
            if (Input.GetKeyDown(KeyCode.R))
            {
                Reset();
            }
            _previousMousePosition = mousePosition;

            if (Time.time < 100.0f)
            {
                AddValueToTexture(
                    _densityOutTexture, 
                    new Vector2(_resolution.x / 2, 1), 
                    new Color(0.00065f, 0.0f, 0.8f, 0.0f) * Time.deltaTime, 
                    3.0f);

                AddValueToTexture(
                    _densityOutTexture, 
                    new Vector2(_resolution.x / 2 - 64, 1), 
                    new Color(0.00055f, 0.0f, 0.6f, 0.0f) * Time.deltaTime, 
                    3.5f);

                AddValueToTexture(
                    _densityOutTexture, 
                    new Vector2(_resolution.x / 2 + 72, 1), 
                    new Color(0.0005f, 0.0f, 0.65f, 0.0f) * Time.deltaTime, 
                    3.0f); 
            }

            // Update compute parameters
            _stableFluids2DCompute.SetFloat("_DeltaTime", Time.deltaTime);

            // Run the fluid simulation
            UpdateVelocity();
            UpdateDensity();
            
            // Update display
            _stableFluids2DCompute.SetTexture(_updateDisplayKernel, "_DisplayTexture", _displayTexture);
            _stableFluids2DCompute.SetTexture(_updateDisplayKernel, "_XIn", _densityOutTexture);
            _stableFluids2DCompute.Dispatch(_updateDisplayKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Add value around given position in target texture.
        /// </summary>
        private void AddValueToTexture(RenderTexture target, Vector2 position, Vector4 value, float radius)
        {
            _stableFluids2DCompute.SetFloat("_AddRadius", radius);
            _stableFluids2DCompute.SetFloats("_AddPosition", new float[2] { position.x, position.y });
            _stableFluids2DCompute.SetVector("_AddValue", value);
            _stableFluids2DCompute.SetTexture(_addValueKernel, "_XOut", target);
            _stableFluids2DCompute.Dispatch(_addValueKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Diffuse and advect density.
        /// </summary>
        private void UpdateDensity()
        {
            Graphics.CopyTexture(_densityOutTexture, _densityInTexture);  // Swap output to input
            Advect(_densityInTexture, _densityOutTexture);

            Graphics.CopyTexture(_densityOutTexture, _densityInTexture);  // Swap output to input
            _stableFluids2DCompute.SetTexture(_updateThermoKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids2DCompute.SetTexture(_updateThermoKernel, "_Thermo", _densityInTexture);
            _stableFluids2DCompute.SetTexture(_updateThermoKernel, "_XOut", _densityOutTexture);
            _stableFluids2DCompute.Dispatch(_updateThermoKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }
        
        /// <summary>
        /// Diffuse and advect velocity, maintaining fluid incompressibility.
        /// </summary>
        private void UpdateVelocity()
        {
            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Swap output to input
            Advect(_velocityInTexture, _velocityOutTexture, true);

            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Swap output to input
            _stableFluids2DCompute.SetTexture(_applyForcesKernel, "_Velocity", _velocityInTexture);
            _stableFluids2DCompute.SetTexture(_applyForcesKernel, "_Thermo", _densityOutTexture);
            _stableFluids2DCompute.SetTexture(_applyForcesKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids2DCompute.SetTexture(_applyForcesKernel, "_XOut", _velocityOutTexture);
            _stableFluids2DCompute.Dispatch(_applyForcesKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);

            ProjectVelocity();
        }

        /// <summary>
        /// Advect input texture by velocity.
        /// </summary>
        private void Advect(RenderTexture inTexture, RenderTexture outTexture, bool setBounds = false)
        {
            // Copy _velocityInTexture to a temporary buffer so that we do not bind it to both
            // _Velocity and _XIn buffers at the same time, since that breaks simulation on Windows machines for some reason...
            Graphics.CopyTexture(_velocityInTexture, _velocityTempTexture);
            
            _stableFluids2DCompute.SetTexture(_advectionKernel, "_Velocity", _velocityTempTexture);
            _stableFluids2DCompute.SetTexture(_advectionKernel, "_XIn", inTexture);
            _stableFluids2DCompute.SetTexture(_advectionKernel, "_XOut", outTexture);
            _stableFluids2DCompute.Dispatch(_advectionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            if (setBounds) SetBounds(inTexture, outTexture);
        }

        /// <summary>
        /// Correct the velocities such that pressure is zero.
        /// </summary>
        private void ProjectVelocity()
        {
            // Projection Part 1
            _stableFluids2DCompute.SetTexture(_projectionPt1Kernel, "_PressureOut", _pressureOutTexture);
            _stableFluids2DCompute.SetTexture(_projectionPt1Kernel, "_Divergence", _divergenceTexture);
            _stableFluids2DCompute.SetTexture(_projectionPt1Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids2DCompute.Dispatch(_projectionPt1Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
            
            // Projection Pt2
            for (int k = 0; k < 5; k++)
            {
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_Divergence", _divergenceTexture);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureOutTexture);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureInTexture);
                _stableFluids2DCompute.Dispatch(_projectionPt2Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureInTexture);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureOutTexture);
                _stableFluids2DCompute.Dispatch(_projectionPt2Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            }

            // Projection Pt3
            _stableFluids2DCompute.SetTexture(_projectionPt3Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids2DCompute.SetTexture(_projectionPt3Kernel, "_PressureIn", _pressureOutTexture);
            _stableFluids2DCompute.Dispatch(_projectionPt3Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
        }
        
        /// <summary>
        /// Enforces boundary condition around edges.
        /// </summary>
        private void SetBounds(RenderTexture inTexture, RenderTexture outTexture)
        {
            _stableFluids2DCompute.SetTexture(_setBoundsXKernel, "_XIn", outTexture);
            _stableFluids2DCompute.SetTexture(_setBoundsXKernel, "_XOut", inTexture);
            _stableFluids2DCompute.Dispatch(_setBoundsXKernel, _setBoundsXThreadCount, 1, 1);
            _stableFluids2DCompute.SetTexture(_setBoundsYKernel, "_XIn", inTexture);
            _stableFluids2DCompute.SetTexture(_setBoundsYKernel, "_XOut", outTexture);
            _stableFluids2DCompute.Dispatch(_setBoundsYKernel, _setBoundsYThreadCount, 1, 1);
        }

        /// <summary>
        /// Reset the simulation.
        /// </summary>
        private void Reset()
        {
            // Initialize "density" textures with initial image
            Graphics.Blit(_initialImage, _densityInTexture);
            Graphics.Blit(_initialImage, _densityOutTexture);
            
            // Initialize velocity to zero
            Graphics.Blit(Texture2D.blackTexture, _velocityInTexture);
            Graphics.Blit(Texture2D.blackTexture, _velocityOutTexture);

            _stableFluids2DCompute.SetTexture(_atmoLUTKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids2DCompute.Dispatch(_atmoLUTKernel, _atmoLUTThreads, 1, 1);

            _stableFluids2DCompute.SetTexture(_resetThermoKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids2DCompute.SetTexture(_resetThermoKernel, "_Thermo", _densityOutTexture);
            _stableFluids2DCompute.Dispatch(_resetThermoKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Blit display texture to screen at end of render pipeline.
        /// </summary>
        void OnEndContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            foreach (Camera cam in cameras)
            {
                Graphics.Blit(_displayTexture, cam.activeTexture);
            }
        }

        private void OnDestroy()
        {
            RenderPipelineManager.endContextRendering -= OnEndContextRendering;
        }
    }
}
