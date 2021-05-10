

IND_LAT=0
IND_LON=1
IND_ALT=2
IND_FC=3
INT_SOW=4

''' Approximate forward map using (reduced) Doppler shift data '''
def fmap_doppler(x, eph, gps_time, acq):
    
    lat = np.radians(x[IND_LAT])
    lon = np.radians(x[IND_LON])
    alt = x[IND_ALT]
    fc = x[IND_FC]
    sow = x[IND_SOW]  # Absolute GPS second of week
    

    doppler_shifts = np.array(acq['doppler'])
    average = np.mean(doppler_shifts)

    max = doppler_magnitudes.maxCoeff()
    height = max - average;
    sigma = 600;
    
    doppler_shifts = doppler_shifts - (fc - acq.get_fc_rough());
    
    error = 0
    sigma_recip = -1.0 / (2.0*sigma*sigma)

    #rp = geography::WGS84(lat,lon,alt);    # Reviever location in ECEF
    
    #for i in range(32):
        #int sv = i+1;
        #satellite s(sv,0,0,0)
        
        #s._eph = eph.get_ephemeris(sv, gt)  # Get the doppler shirts at locaton rp.
        
        #double elevation = elevAngle(rp,s.location(sow))
        #double sim_doppler_shift = s.doppler(sow, rp)

        #VectorXd sim_doppler_magnitudes(nOfBins)
        
        #sim_doppler_magnitudes.setConstant(average)
        #if (elevation>0):
            #double strength = height*signal_strength(elevation);

            #for j in range(nOfBins):
                #double diff = doppler_shifts[j]-sim_doppler_shift;
                #sim_doppler_magnitudes(j) += strength*exp(sigma_recip*diff*diff);
        #VectorXd diff = sim_doppler_magnitudes - doppler_magnitudes.col(i);
        #error += diff.array().square().sum();
        
    
    #return exp(-error)

if __name__=="__main__":
    pass
