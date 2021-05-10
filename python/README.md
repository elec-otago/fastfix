# Steps to calculate:
#
#openocd -f interface/jlink.cfg -f board/olimex_stm32_h103.cfg -c "init" -c "reset" -c "shutdown"

./download_data.sh datafile

#SAMPLING_RATE=4.092e6
SAMPLING_RATE=8.184e6
BIT_RES=1
# View the spectrum of the data
#octave --persist --eval "gps_spectrum('datafile.bin', ${SAMPLING_RATE}, ${BIT_RES})"
# Print it with print('spectrum.pdf','-dpdf')

# Calculate correlation with SV 2
python bit_error_rate.py datafile.bin ${SAMPLING_RATE} ${BIT_RES}

## Test Data

Located in freenas/tag/neo_test_data/roof_test
