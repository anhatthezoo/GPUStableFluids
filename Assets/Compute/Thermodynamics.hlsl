#define G           9.81                    // gravity (m/s)
#define R_D         287.0                   // gas constant for dry air (J/kg*k)
#define C_P         1003.5                  // specific heat capacity for dry air at constant pressure (J/kg*k)
#define L           2501000.0               // latent heat of vaporization of water (kj/kg)
#define THETA_V0    295.0                   // reference potential temperature (k)

float _VolumeHeight;
float _GroundTemperature;
float _GroundPressure;
float _LapseRate;

float KtoC(float k) { return k - 273.15; }

float CtoK(float c) { return c + 273.15; }

float getHeightInVolume(uint layer) 
{
    return ((float)layer / (float)(_Resolution.y - 1)) * _VolumeHeight;
}

float getAbsoluteTemperature(float h) 
{
    return _GroundTemperature - (_LapseRate * h);
}

float getAirPressure(float h) 
{
    float exponent = G / (_LapseRate * R_D);
    return _GroundPressure * pow(1.0 - ((h * _LapseRate) / _GroundTemperature), exponent);
}

float exner(float p) 
{
    float exponent = R_D / C_P;
    return pow(p / _GroundPressure, exponent);
}

float getSaturationMixingRatio(float T, float p)
{
    return (380.16 / p) * exp((17.67 * KtoC(T)) / (KtoC(T) + 243.5));
}

float2 getAmbientThermoState(uint layer) 
{
    float2 tempPressure = _AtmoLUT[uint2(layer, 0)].rg;
    float localTemp = tempPressure.x;
    float airPressure = tempPressure.y;

    float theta = localTemp / exner(airPressure);
    float q_v;

    float normalizedAltitude = (float)layer / (float)(_Resolution.y - 1);

    if (normalizedAltitude < 0.3) {
        q_v = getSaturationMixingRatio(localTemp - 2.0, airPressure);
    } else {
        q_v = getSaturationMixingRatio(localTemp - 20.0, airPressure);
    }

    return float2(q_v, theta);
}