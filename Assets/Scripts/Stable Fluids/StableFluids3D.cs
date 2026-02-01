using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace StableFluids
{
    /// <summary>
    /// A GPU implementation of Jos Stam's Stable Fluids in 3D.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer))]
    public class StableFluids3D : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField] private ComputeShader _stableFluids3DCompute;
        [SerializeField] private Material _volumeMaterial;

        [Header("Fluid Simulation Parameters")] 
        [SerializeField] private Vector3Int _resolution;
        [SerializeField] private Vector3Int _volumeSize;
        [SerializeField] private float _diffusion;
        [SerializeField] private float _vorticity;
        [SerializeField] private float _groundTemperature;
        [SerializeField] private float _groundPressure;
        [SerializeField] private float _lapseRate;

        [Header("Source Parameters")]
        [SerializeField] private int _sourceCount = 12;
        [SerializeField] private float _sourceRadiusMin = 3f;
        [SerializeField] private float _sourceRadiusMax = 6f;
        [SerializeField] private float _moistureMin = 0.0008f;
        [SerializeField] private float _moistureMax = 0.002f;
        [SerializeField] private float _heatMin = 0.8f;
        [SerializeField] private float _heatMax = 2.0f;
        [SerializeField] private float _sourceDuration = 60f;

        [Header("Wind Parameters")]
        [SerializeField] private float _windSpeedKnots = 10f;
        [SerializeField] [Range(0f, 360f)] private float _windDirectionDegrees = 270f;  // 0=N, 90=E, 180=S, 270=W
        [SerializeField] [Range(0f, 5f)] private float _windShearExponent = 3f;  // How much wind increases with height

        // Knots to meters per second
        private const float KNOTS_TO_MS = 0.514444f;

        // Moisture source data - matches GPU struct layout
        private struct GPUSource
        {
            public Vector3 position;
            public float radius;
            public Vector3 value;  // moisture, 0, heat
        }

        // Source configuration (CPU side)
        private struct SourceConfig
        {
            public Vector3 position;
            public float radius;
            public float moisture;
            public float heat;
            public float phaseOffset;
        }
        private SourceConfig[] _sourceConfigs;
        private GPUSource[] _gpuSources;
        private ComputeBuffer _sourceBuffer;

        // Input
        private Vector2 _previousMousePosition;
        
        // Mesh renderer
        private MeshRenderer _meshRenderer;

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
        private int _clearKernel;
        private int _updateDisplayKernel;
        private int _atmoLUTKernel;
        private int _resetThermoKernel;
        private int _addValueKernel;
        private int _addSourcesBatchedKernel;
        private int _advectionKernel;
        private int _applyForcesKernel;
        private int _updateThermoKernel;
        private int _projectionPt1Kernel;
        private int _projectionPt2Kernel;
        private int _projectionPt3Kernel;
        private int _setBoundsXYKernel;
        private int _setBoundsYZKernel;
        private int _setBoundsZXKernel;

        // Thread groups
        private Vector3Int _threadGroups;
        private int _atmoLUTThreads;
        private Vector3Int _setBoundsXYThreadGroups;
        private Vector3Int _setBoundsYZThreadGroups;
        private Vector3Int _setBoundsZXThreadGroups;

        private void Start()
        {
            // Setup mesh renderer
            _meshRenderer = GetComponent<MeshRenderer>();
            _meshRenderer.material = _volumeMaterial;

            // Get kernel ids
            _clearKernel = _stableFluids3DCompute.FindKernel("Clear");
            _updateDisplayKernel = _stableFluids3DCompute.FindKernel("UpdateDisplay");
            _atmoLUTKernel = _stableFluids3DCompute.FindKernel("GenerateAtmoLUT");
            _resetThermoKernel = _stableFluids3DCompute.FindKernel("ResetThermodynamics");
            _addValueKernel = _stableFluids3DCompute.FindKernel("AddValue");
            _addSourcesBatchedKernel = _stableFluids3DCompute.FindKernel("AddSourcesBatched");
            _advectionKernel = _stableFluids3DCompute.FindKernel("Advection");
            _applyForcesKernel = _stableFluids3DCompute.FindKernel("ApplyForces");
            _updateThermoKernel = _stableFluids3DCompute.FindKernel("UpdateThermodynamics");
            _projectionPt1Kernel = _stableFluids3DCompute.FindKernel("ProjectionPt1");
            _projectionPt2Kernel = _stableFluids3DCompute.FindKernel("ProjectionPt2");
            _projectionPt3Kernel = _stableFluids3DCompute.FindKernel("ProjectionPt3");
            _setBoundsXYKernel = _stableFluids3DCompute.FindKernel("SetBoundsXY");
            _setBoundsYZKernel = _stableFluids3DCompute.FindKernel("SetBoundsYZ");
            _setBoundsZXKernel = _stableFluids3DCompute.FindKernel("SetBoundsZX");
            
            // Get thread counts
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_advectionKernel, out uint xThreadGroupSize, out uint yThreadGroupSize, out uint zThreadGroupSize);
            _threadGroups = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z / (float)zThreadGroupSize));
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_atmoLUTKernel,
                out xThreadGroupSize, out _, out _);
            _atmoLUTThreads = Mathf.CeilToInt(_resolution.y / (float)xThreadGroupSize);
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_setBoundsXYKernel, out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsXYThreadGroups = new Vector3Int(
                Mathf.CeilToInt(_resolution.x * 2 / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(1 / (float)zThreadGroupSize));
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_setBoundsYZKernel, out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsYZThreadGroups = new Vector3Int(
                Mathf.CeilToInt(1 / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y * 2/ (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z / (float)zThreadGroupSize));
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_setBoundsZXKernel, out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsZXThreadGroups = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(1 / (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z * 2 / (float)zThreadGroupSize));

            // Pass constants to compute
            _stableFluids3DCompute.SetInts("_Resolution", new int[] { _resolution.x, _resolution.y, _resolution.z });
            _stableFluids3DCompute.SetFloat("_VolumeHeight", _volumeSize.y);
            _stableFluids3DCompute.SetFloat("_Vorticity", _vorticity);
            _stableFluids3DCompute.SetFloat("_GroundTemperature", _groundTemperature);
            _stableFluids3DCompute.SetFloat("_GroundPressure", _groundPressure);
            _stableFluids3DCompute.SetFloat("_LapseRate", _lapseRate);

            // Allocate render textures
            _displayTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 4);
            _densityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 4);
            _densityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 4);
            _velocityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _velocityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _velocityTempTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _pressureOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 1);
            _pressureInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 1);
            _divergenceTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 1);
            _atmosphereLUT = RenderTextureUtilities.AllocateRWLinearRT(_resolution.y, 1, 0, 2);
            
            // Pass params to volume rendering material
            _volumeMaterial.SetTexture("_Voxels", _displayTexture);
            _volumeMaterial.SetVector("_Dimensions", (Vector3)_resolution);
            
            // Reset simulation
            Reset();
        }

        /// <summary>
        /// Process input and run fluid simulation.
        /// </summary>
        void Update()
        {
            // Input
            if (Input.GetKeyDown(KeyCode.R))
            {
                Reset();
            }
            
            // Add moisture and heat sources on the ground (batched - single dispatch)
            if (_sourceBuffer != null && Time.time < _sourceDuration)
            {
                // Update GPU source values with time variation
                for (int i = 0; i < _sourceConfigs.Length; i++)
                {
                    var cfg = _sourceConfigs[i];
                    float timeVar = 0.7f + 0.3f * Mathf.Sin(Time.time * 0.5f + cfg.phaseOffset);

                    _gpuSources[i].position = cfg.position;
                    _gpuSources[i].radius = cfg.radius;
                    _gpuSources[i].value = new Vector3(
                        cfg.moisture * timeVar,
                        0f,
                        cfg.heat * timeVar
                    ) * Time.deltaTime;
                }

                // Upload and dispatch once
                _sourceBuffer.SetData(_gpuSources);
                _stableFluids3DCompute.SetBuffer(_addSourcesBatchedKernel, "_Sources", _sourceBuffer);
                _stableFluids3DCompute.SetInt("_SourceCount", _sourceConfigs.Length);
                _stableFluids3DCompute.SetTexture(_addSourcesBatchedKernel, "_XOut", _densityOutTexture);
                _stableFluids3DCompute.Dispatch(_addSourcesBatchedKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            }

            // Update compute parameters
            _stableFluids3DCompute.SetFloat("_DeltaTime", Time.deltaTime);

            // Calculate wind velocity in simulation units
            // Convert knots to m/s, then to cells/second based on volume/resolution ratio
            float windSpeedMS = _windSpeedKnots * KNOTS_TO_MS;
            float cellsPerMeterX = _resolution.x / (float)_volumeSize.x;
            float cellsPerMeterZ = _resolution.z / (float)_volumeSize.z;

            // Wind direction: 0=N(+Z), 90=E(+X), 180=S(-Z), 270=W(-X)
            float windRadians = _windDirectionDegrees * Mathf.Deg2Rad;
            Vector3 windVelocity = new Vector3(
                Mathf.Sin(windRadians) * windSpeedMS * cellsPerMeterX,
                0f,
                Mathf.Cos(windRadians) * windSpeedMS * cellsPerMeterZ
            );

            _stableFluids3DCompute.SetVector("_WindVelocity", windVelocity);
            _stableFluids3DCompute.SetFloat("_WindShearExponent", _windShearExponent);

            // Run the fluid simulation
            UpdateVelocity();
            UpdateDensity();
            
            // Update display
            _stableFluids3DCompute.SetTexture(_updateDisplayKernel, "_DisplayTexture", _displayTexture);
            _stableFluids3DCompute.SetTexture(_updateDisplayKernel, "_XIn", _densityOutTexture);
            _stableFluids3DCompute.Dispatch(_updateDisplayKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
        }

        /// <summary>
        /// Add value around given position in target texture.
        /// </summary>
        private void AddValueToTexture(RenderTexture target, Vector3 position, Vector4 value, float radius)
        {
            _stableFluids3DCompute.SetFloat("_AddRadius", radius);
            _stableFluids3DCompute.SetVector("_AddPosition", position);
            _stableFluids3DCompute.SetVector("_AddValue", value);
            _stableFluids3DCompute.SetTexture(_addValueKernel, "_XOut", target);
            _stableFluids3DCompute.Dispatch(_addValueKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
        }

        /// <summary>
        /// Diffuse and advect density.
        /// </summary>
        private void UpdateDensity()
        {
            Graphics.CopyTexture(_densityOutTexture, _densityInTexture);  // Swap output to input
            Advect(_densityInTexture, _densityOutTexture);

            Graphics.CopyTexture(_densityOutTexture, _densityInTexture);  // Swap output to input
            _stableFluids3DCompute.SetTexture(_updateThermoKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids3DCompute.SetTexture(_updateThermoKernel, "_Thermo", _densityInTexture);
            _stableFluids3DCompute.SetTexture(_updateThermoKernel, "_XOut", _densityOutTexture);
            _stableFluids3DCompute.Dispatch(_updateThermoKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
        }
        
        /// <summary>
        /// Diffuse and advect velocity, maintaining fluid incompressibility.
        /// </summary>
        private void UpdateVelocity()
        {
            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Swap output to input
            Advect(_velocityInTexture, _velocityOutTexture, true);

            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Swap output to input
            _stableFluids3DCompute.SetTexture(_applyForcesKernel, "_Velocity", _velocityInTexture);
            _stableFluids3DCompute.SetTexture(_applyForcesKernel, "_Thermo", _densityOutTexture);
            _stableFluids3DCompute.SetTexture(_applyForcesKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids3DCompute.SetTexture(_applyForcesKernel, "_XOut", _velocityOutTexture);
            _stableFluids3DCompute.Dispatch(_applyForcesKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);

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
            
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_Velocity", _velocityTempTexture);
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_XIn", inTexture);
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_XOut", outTexture);
            _stableFluids3DCompute.Dispatch(_advectionKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            if (setBounds) SetBounds(inTexture, outTexture);
        }

        /// <summary>
        /// Correct the velocities such that pressure is zero.
        /// </summary>
        private void ProjectVelocity()
        {
            // Projection Part 1
            _stableFluids3DCompute.SetTexture(_projectionPt1Kernel, "_PressureOut", _pressureOutTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt1Kernel, "_Divergence", _divergenceTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt1Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids3DCompute.Dispatch(_projectionPt1Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
            
            // Projection Pt2
            for (int k = 0; k < 5; k++)
            {
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_Divergence", _divergenceTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureOutTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureInTexture);
                _stableFluids3DCompute.Dispatch(_projectionPt2Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureInTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureOutTexture);
                _stableFluids3DCompute.Dispatch(_projectionPt2Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            }

            // Projection Pt3
            _stableFluids3DCompute.SetTexture(_projectionPt3Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt3Kernel, "_PressureIn", _pressureOutTexture);
            _stableFluids3DCompute.Dispatch(_projectionPt3Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
        }
        
        /// <summary>
        /// Enforces boundary condition around faces.
        /// </summary>
        private void SetBounds(RenderTexture inTexture, RenderTexture outTexture)
        { 
            _stableFluids3DCompute.SetTexture(_setBoundsXYKernel, "_XIn", outTexture);
            _stableFluids3DCompute.SetTexture(_setBoundsXYKernel, "_XOut", inTexture);
            _stableFluids3DCompute.Dispatch(_setBoundsXYKernel, _setBoundsXYThreadGroups.x, _setBoundsXYThreadGroups.y, _setBoundsXYThreadGroups.z);
            _stableFluids3DCompute.SetTexture(_setBoundsYZKernel, "_XIn", inTexture);
            _stableFluids3DCompute.SetTexture(_setBoundsYZKernel, "_XOut", outTexture);
            _stableFluids3DCompute.Dispatch(_setBoundsYZKernel, _setBoundsYZThreadGroups.x, _setBoundsYZThreadGroups.y, _setBoundsYZThreadGroups.z);
            _stableFluids3DCompute.SetTexture(_setBoundsZXKernel, "_XIn", outTexture);
            _stableFluids3DCompute.SetTexture(_setBoundsZXKernel, "_XOut", inTexture);
            _stableFluids3DCompute.Dispatch(_setBoundsZXKernel, _setBoundsZXThreadGroups.x, _setBoundsZXThreadGroups.y, _setBoundsZXThreadGroups.z);
        }

        /// <summary>
        /// Reset the simulation.
        /// </summary>
        private void Reset()
        {
            // Generate new random sources
            GenerateSources();

            // Initialize velocity to zero
            _stableFluids3DCompute.SetTexture(_clearKernel, "_XOut", _velocityInTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            _stableFluids3DCompute.SetTexture(_clearKernel, "_XOut", _velocityOutTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);

            _stableFluids3DCompute.SetTexture(_atmoLUTKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids3DCompute.Dispatch(_atmoLUTKernel, _atmoLUTThreads, 1, 1);

            _stableFluids3DCompute.SetTexture(_resetThermoKernel, "_AtmoLUT", _atmosphereLUT);
            _stableFluids3DCompute.SetTexture(_resetThermoKernel, "_Thermo", _densityOutTexture);
            _stableFluids3DCompute.Dispatch(_resetThermoKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
        }

        /// <summary>
        /// Generate random moisture/heat sources scattered across the ground.
        /// </summary>
        private void GenerateSources()
        {
            // Release old buffer if it exists
            _sourceBuffer?.Release();

            _sourceConfigs = new SourceConfig[_sourceCount];
            _gpuSources = new GPUSource[_sourceCount];

            // Margin from edges to avoid boundary issues
            float margin = 10f;

            for (int i = 0; i < _sourceCount; i++)
            {
                // Bias toward stronger sources for cumulonimbus formation
                // Using squared random gives more weight to higher values
                float intensity = Random.value;
                intensity = intensity * intensity;  // Bias toward high values

                // Stronger sources get larger radius, more moisture, more heat
                float radius = Mathf.Lerp(_sourceRadiusMin, _sourceRadiusMax * 1.0f, intensity);
                float moisture = Mathf.Lerp(_moistureMin, _moistureMax * 1.0f, intensity);
                float heat = Mathf.Lerp(_heatMin, _heatMax * 1.0f, intensity);

                _sourceConfigs[i] = new SourceConfig
                {
                    position = new Vector3(
                        Random.Range(margin, _resolution.x - margin),
                        1f,  // Ground level
                        Random.Range(margin, _resolution.z - margin)
                    ),
                    radius = radius,
                    moisture = moisture,
                    heat = heat,
                    phaseOffset = Random.Range(0f, Mathf.PI * 2f)
                };
            }

            // Create compute buffer for GPU sources (position: 3 floats, radius: 1 float, value: 3 floats = 7 floats = 28 bytes)
            _sourceBuffer = new ComputeBuffer(_sourceCount, sizeof(float) * 7);
        }

        private void OnDestroy()
        {
            _sourceBuffer?.Release();
        }
    }
}
